"""Market and desk inputs for the jump-to-default (JTD) PDE model.

Kept separate from the TF ``MarketData`` because the two models consume
credit differently: TF takes a single credit spread and discounts the
cash-only leg at ``r + s``; the JTD model takes a *default intensity*
``λ(S, t)`` that is stock-dependent and pays recovery on default.

All rates, yields and intensities are continuously compounded and
annualized; times are ACT/365 year fractions from the valuation date,
matching the TF side (docs/MODEL.md).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RecoveryConvention(Enum):
    """What the bondholder receives at default (spec §02).

    - ``PAR``: ``R · N`` — the v1 default.
    - ``PAR_PLUS_ACCRUED``: ``R · (N + A(t))`` — claim includes accrued.
    - ``MARKET_VALUE``: ``R · V`` — recovery of pre-default market value.
      Implemented linearly (folded into the discount rate), which
      ignores conversion into post-default shares; exact only for
      ``η = 1``, where post-default shares are worthless anyway.
    """

    PAR = "par"
    PAR_PLUS_ACCRUED = "par_plus_accrued"
    MARKET_VALUE = "market_value"


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-constant base intensity ``λ0(t)``, anchored at a reference spot.

    ``λ0`` equals ``intensities[j]`` on ``(pillars[j-1], pillars[j]]``
    (with ``pillars[-1] := 0``) and is extrapolated flat beyond the
    last pillar.

    ``reference_spot`` is the ``S0`` in ``λ(S, t) = λ0(t) (S0/S)^p``.
    It is fixed at calibration time and deliberately does *not* move
    when spot is bumped: a spot move then changes the hazard along
    ``λ(S)``, which is the intended "credit delta" (spec §07, sticky
    assumptions).
    """

    pillars: tuple[float, ...]
    intensities: tuple[float, ...]
    reference_spot: float

    def __post_init__(self) -> None:
        if len(self.pillars) != len(self.intensities) or not self.pillars:
            raise ValueError(
                "pillars and intensities must be non-empty and equal length"
            )
        if any(b <= a for a, b in zip(self.pillars, self.pillars[1:])):
            raise ValueError("pillars must be strictly increasing")
        if any(x < 0 for x in self.intensities):
            raise ValueError("intensities must be non-negative")

    @classmethod
    def flat(cls, intensity: float, reference_spot: float) -> "HazardCurve":
        """Constant ``λ0`` for all t."""
        return cls(
            pillars=(1.0,), intensities=(intensity,), reference_spot=reference_spot
        )

    def base_intensity(self, t: float) -> float:
        """``λ0(t)`` with right-closed buckets ``(T_{j-1}, T_j]``."""
        j = int(np.searchsorted(np.asarray(self.pillars), t - 1e-12, side="left"))
        return self.intensities[min(j, len(self.intensities) - 1)]

    def shifted(self, amount: float) -> "HazardCurve":
        """Parallel shift of every ``λ0`` bucket, floored at zero."""
        return HazardCurve(
            self.pillars,
            tuple(max(x + amount, 0.0) for x in self.intensities),
            self.reference_spot,
        )

    def scaled(self, factor: float) -> "HazardCurve":
        return HazardCurve(
            self.pillars,
            tuple(x * factor for x in self.intensities),
            self.reference_spot,
        )


@dataclass(frozen=True)
class CDSCurve:
    """Market CDS par spreads (decimal, e.g. 0.018 = 180bp) at pillar maturities."""

    pillars: tuple[float, ...]
    spreads: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.pillars) != len(self.spreads) or not self.pillars:
            raise ValueError("pillars and spreads must be non-empty and equal length")
        if any(b <= a for a, b in zip(self.pillars, self.pillars[1:])):
            raise ValueError("pillars must be strictly increasing")

    def shifted(self, amount: float, pillar: int | None = None) -> "CDSCurve":
        """Parallel shift, or shift a single pillar when ``pillar`` is given."""
        spreads = tuple(
            s + amount if pillar is None or j == pillar else s
            for j, s in enumerate(self.spreads)
        )
        return CDSCurve(self.pillars, spreads)


@dataclass(frozen=True)
class JTDMarketData:
    """Everything the JTD engine needs besides the bond terms.

    ``hazard`` is always what the PDE uses. ``cds_curve`` is optional
    provenance: when present, ``hazard`` is assumed to have been
    bootstrapped from it, and Greeks that bump rates, recovery or the
    CDS curve re-bootstrap ``hazard`` with spreads held fixed (spec §07).
    Without it, the hazard curve is treated as the market input.

    ``volatility`` is the *diffusion* vol of this model — not a
    Black–Scholes implied vol, which already contains default risk and
    would double-count it (spec §05).

    ``borrow_cost`` acts exactly like an extra dividend yield in the
    drift. ``discrete_dividends`` are ``(time, cash amount per share)``
    and are applied as jump conditions on top of ``dividend_yield``.
    """

    spot: float
    volatility: float
    risk_free_rate: float
    hazard: HazardCurve
    hazard_elasticity: float
    dividend_yield: float = 0.0
    borrow_cost: float = 0.0
    discrete_dividends: tuple[tuple[float, float], ...] = ()
    recovery: float = 0.4
    jump_size: float = 1.0
    lambda_max: float = 2.0
    recovery_convention: RecoveryConvention = RecoveryConvention.PAR
    cds_curve: CDSCurve | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.jump_size <= 1.0:
            raise ValueError("jump_size η must be in (0, 1]")
        if self.hazard_elasticity < 0:
            raise ValueError("hazard_elasticity p must be >= 0")
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError("recovery must be in [0, 1)")

    def intensity(self, spots: np.ndarray, t: float) -> np.ndarray:
        """``λ(S, t) = min(λ0(t) (S_ref/S)^p, λmax)``, safe at ``S = 0``."""
        lam0 = self.hazard.base_intensity(t)
        spots = np.asarray(spots, dtype=float)
        if lam0 == 0.0:
            return np.zeros_like(spots)
        p = self.hazard_elasticity
        if p == 0.0:
            return np.full_like(spots, min(lam0, self.lambda_max))
        with np.errstate(divide="ignore", over="ignore"):
            ratio = np.where(
                spots > 0,
                self.hazard.reference_spot / np.maximum(spots, 1e-300),
                np.inf,
            )
            return np.minimum(lam0 * ratio**p, self.lambda_max)
