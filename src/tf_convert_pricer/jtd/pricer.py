"""Convertible bond pricer under the jump-to-default PDE (spec §02–§04).

Pieces, each small enough to test on its own:

  - ``JTDTerms``: term-sheet features the TF ``ConvertibleBond`` doesn't
    carry (soft-call trigger, notice period, clean/dirty and accrued
    conventions). Kept out of ``ConvertibleBond`` so the TF pricer
    doesn't silently ignore fields it can't model.
  - ``build_grid``: space/time grid with every event on a node.
  - ``solve_grid``: one backward PDE solve → values at ``t0`` on the grid.
  - ``price``: the per-valuation outputs (§01) — dirty/clean price,
    parity, premiums, bond floor, grid delta and gamma.

Schedule semantics on ``ConvertibleBond`` as used here:
  - ``call_schedule`` keeps TF's step-function meaning: ``{t: Bc}`` is
    "callable at ``Bc`` from ``t`` until the next entry or maturity".
  - ``put_schedule`` is **Bermudan**: ``{t: Bp}`` is "puttable at ``Bp``
    on date ``t`` only" (spec §03). This differs from the TF pricer,
    which treats puts as step functions too.
  - Call and put prices are clean by default (``JTDTerms.prices_are_clean``)
    and have accrued added at exercise.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import CubicSpline

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.jtd.analytic import black_scholes_call
from tf_convert_pricer.jtd.grid import (
    GridSpec,
    node_index,
    smax_for,
    space_grid,
    time_grid,
)
from tf_convert_pricer.jtd.market import JTDMarketData, RecoveryConvention
from tf_convert_pricer.jtd.pde import solve_backward
from tf_convert_pricer.pricer import coupon_schedule


@dataclass(frozen=True)
class JTDTerms:
    """Call/put/conversion conventions beyond the core bond terms.

    ``call_trigger`` is the soft-call ``h``: the bond is callable at ``t``
    only where ``S ≥ h · N/κ``. ``h = 0`` is a hard call. This is the
    spec's Markov approximation of an "``m``-of-``n`` days" trigger;
    raise ``h`` slightly to proxy the averaging requirement.

    ``call_notice`` is ``n`` in years: once called, the holder keeps the
    cash-or-shares choice until ``t + n`` (default during notice ignored).

    ``exercise_before_coupon`` decides what happens *on* a coupon date:
      - ``True`` (default): call / put / conversion apply cum-coupon, with
        the full coupon as accrued — a holder forced to convert forfeits
        it (issuers time calls to exploit this). This is the TF tree's
        ordering and the one that reproduces the spec's reference run.
      - ``False``: the coupon is paid first, then call / put / conversion
        apply ex-coupon with zero accrued.
    Between coupon dates the two agree up to O(Δt); they differ by up to
    a whole coupon when a call period *starts* on a coupon date.
    """

    call_trigger: float = 0.0
    call_notice: float = 0.0
    prices_are_clean: bool = True
    conversion_forfeits_accrued: bool = True
    exercise_before_coupon: bool = True


@dataclass(frozen=True)
class PricingGrid:
    spots: np.ndarray
    spot_index: int
    times: np.ndarray

    def rolled(self, dt: float) -> "PricingGrid":
        """Same space grid, time grid seen from ``t0 + dt`` (for theta)."""
        later = self.times[self.times > dt + 1e-9] - dt
        return PricingGrid(self.spots, self.spot_index, np.concatenate(([0.0], later)))


def _conversion_price(bond: ConvertibleBond) -> float:
    return (
        bond.face_value / bond.conversion_ratio if bond.conversion_ratio > 0 else np.inf
    )


def _redemption(bond: ConvertibleBond) -> float:
    return (
        bond.redemption_value if bond.redemption_value is not None else bond.face_value
    )


def event_times(
    bond: ConvertibleBond, market: JTDMarketData
) -> tuple[list[float], list[float]]:
    """``(restart_events, all_grid_events)``.

    Restart events are the non-smooth ones that get a Rannacher restart;
    hazard pillars only need to be nodes (so ``λ0`` is constant per step).
    """
    restarts = [t for t, _ in coupon_schedule(bond)]
    restarts += [t for t, _ in market.discrete_dividends]
    restarts += list((bond.call_schedule or {}).keys())
    restarts += list((bond.put_schedule or {}).keys())
    return restarts, restarts + list(market.hazard.pillars)


def build_grid(
    bond: ConvertibleBond,
    market: JTDMarketData,
    spec: GridSpec = GridSpec(),
    terms: "JTDTerms | None" = None,
) -> PricingGrid:
    """Grid for ``bond``; pass ``terms`` so a soft-call trigger is pinned to a node."""
    strike = _conversion_price(bond)
    if not np.isfinite(strike):
        strike = market.spot
    smax = smax_for(market.spot, strike, market.volatility, bond.maturity, spec)
    pins = (strike,) if np.isfinite(_conversion_price(bond)) else ()
    if (
        terms is not None
        and terms.call_trigger > 0
        and np.isfinite(_conversion_price(bond))
    ):
        pins += (terms.call_trigger * strike,)
    spots, j0 = space_grid(market.spot, smax, spec, pins)
    _, events = event_times(bond, market)
    return PricingGrid(spots, j0, time_grid(bond.maturity, events, spec.steps_per_year))


def accrued_interest(bond: ConvertibleBond) -> Callable[[float], float]:
    """``A(t)``, right-continuous: zero *at* a coupon date (just paid).

    Linear accrual over ``(t_i − 1/f, t_i]``, measured from the previous
    scheduled date even when that date is before the valuation date.
    """
    coupons = coupon_schedule(bond)
    times = np.array([t for t, _ in coupons])
    amounts = [a for _, a in coupons]
    period = 1.0 / bond.coupon_frequency

    def accrued(t: float) -> float:
        i = int(np.searchsorted(times, t + 1e-9, side="left"))
        if i >= len(times):
            return 0.0
        return amounts[i] * max(t - (times[i] - period), 0.0) / period

    return accrued


def _effective_call_price(
    dirty_call: float,
    spots: np.ndarray,
    bond: ConvertibleBond,
    terms: JTDTerms,
    market: JTDMarketData,
) -> np.ndarray:
    """``B̃c_eff(S) = e^{−rn} B̃c + κ C_BS(S, B̃c/κ, n)``; collapses to ``B̃c`` at ``n = 0``.

    The BS call uses this model's diffusion vol and ``q + b`` as the
    yield, and ignores default during the notice window (spec §04).
    """
    n = terms.call_notice
    if n <= 0.0:
        return np.full_like(spots, dirty_call)
    kappa = bond.conversion_ratio
    cash = np.exp(-market.risk_free_rate * n) * dirty_call
    if kappa <= 0.0:
        return np.full_like(spots, cash)
    option = black_scholes_call(
        spots,
        dirty_call / kappa,
        n,
        market.risk_free_rate,
        market.volatility,
        market.dividend_yield + market.borrow_cost,
    )
    return cash + kappa * np.asarray(option)


def _active_calls(bond: ConvertibleBond, times: np.ndarray) -> dict[int, float]:
    """Clean call price active at each time index (step-function schedule)."""
    if not bond.call_schedule:
        return {}
    events = sorted(bond.call_schedule.items())
    out: dict[int, float] = {}
    for k, t in enumerate(times[:-1]):
        active = None
        for start, level in events:
            if start <= t + 1e-9:
                active = level
        if active is not None:
            out[k] = active
    return out


def _on_grid(times: np.ndarray, t: float, what: str) -> int:
    k = node_index(times, t)
    if k is None:
        raise ValueError(f"{what} at t={t} is not a node of the time grid")
    return k


def default_value(
    bond: ConvertibleBond, market: JTDMarketData, spots: np.ndarray, accrued: float
) -> np.ndarray:
    """``V_D(S, t)`` for the recovery-of-par conventions.

    ``max(κ(1 − η)S, R · claim)``: the holder converts into post-default
    shares if those are worth more than the recovery.
    """
    claim = bond.face_value
    if market.recovery_convention is RecoveryConvention.PAR_PLUS_ACCRUED:
        claim += accrued
    shares = bond.conversion_ratio * (1.0 - market.jump_size) * spots
    return np.maximum(shares, market.recovery * claim)


def solve_grid(
    bond: ConvertibleBond,
    market: JTDMarketData,
    grid: PricingGrid,
    terms: JTDTerms = JTDTerms(),
    rannacher: bool = True,
) -> np.ndarray:
    """Convertible value per bond at ``t0`` on every node of ``grid.spots``."""
    spots, times = grid.spots, grid.times
    maturity = times[-1]
    if abs(maturity - bond.maturity) > 1e-9:
        raise ValueError("time grid does not end at bond maturity")
    kappa = bond.conversion_ratio
    accrued = accrued_interest(bond)
    r = market.risk_free_rate
    carry = r - market.dividend_yield - market.borrow_cost
    eta, rec = market.jump_size, market.recovery
    rmv = market.recovery_convention is RecoveryConvention.MARKET_VALUE

    def coefficients(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lam = market.intensity(spots, t)
        drift = carry + eta * lam
        if rmv:
            return drift, r + lam * (1.0 - rec), np.zeros_like(spots)
        return drift, r + lam, lam * default_value(bond, market, spots, accrued(t))

    coupons: dict[int, float] = {}
    final_coupon = 0.0
    for t, amount in coupon_schedule(bond):
        if abs(t - maturity) < 1e-9:
            final_coupon += amount
        else:
            k = _on_grid(times, t, "coupon")
            coupons[k] = coupons.get(k, 0.0) + amount
    dividends: dict[int, float] = {}
    for t, amount in market.discrete_dividends:
        if 1e-9 < t < maturity - 1e-9:
            k = _on_grid(times, t, "dividend")
            dividends[k] = dividends.get(k, 0.0) + amount
    puts: dict[int, float] = {}
    for t, level in (bond.put_schedule or {}).items():
        if abs(t - maturity) < 1e-9:
            raise ValueError("put at maturity: express it through redemption_value")
        if -1e-9 <= t < maturity:
            puts[_on_grid(times, t, "put date")] = level
    calls = _active_calls(bond, times)
    for t in bond.call_schedule or {}:
        if 1e-9 < t < maturity - 1e-9:
            _on_grid(times, t, "call start")
    restart = (
        frozenset(coupons)
        | frozenset(dividends)
        | frozenset(puts)
        | frozenset(
            node_index(times, t)
            for t in (bond.call_schedule or {})
            if node_index(times, t) is not None
        )
    )

    if kappa > 0:
        trigger_level = terms.call_trigger * bond.face_value / kappa
    else:
        trigger_level = 0.0 if terms.call_trigger == 0.0 else np.inf
    callable_region = spots >= trigger_level - 1e-12

    def conversion_value(acc: float) -> np.ndarray:
        return kappa * spots + (0.0 if terms.conversion_forfeits_accrued else acc)

    def constrain(k: int, values: np.ndarray, acc: float) -> np.ndarray:
        """Issuer call → holder put → holder conversion (spec §04 order)."""
        conv = conversion_value(acc)
        if k in calls:
            dirty = calls[k] + (acc if terms.prices_are_clean else 0.0)
            cap = np.maximum(
                _effective_call_price(dirty, spots, bond, terms, market), conv
            )
            values = np.where(callable_region, np.minimum(values, cap), values)
        if k in puts:
            values = np.maximum(
                values, puts[k] + (acc if terms.prices_are_clean else 0.0)
            )
        if kappa > 0:
            values = np.maximum(values, conv)
        return values

    def on_node(k: int, values: np.ndarray) -> np.ndarray:
        coupon = coupons.get(k, 0.0)
        acc_after = accrued(times[k])
        acc_before = acc_after + coupon
        if coupon and terms.exercise_before_coupon:
            values = constrain(k, values + coupon, acc_before)
        else:
            values = constrain(k, values, acc_after) + coupon
        if k in dividends:
            shifted = np.maximum(spots - dividends[k], 0.0)
            values = CubicSpline(spots, values)(shifted)
        if (coupon or k in dividends) and kappa > 0:
            values = np.maximum(values, conversion_value(acc_before))
        return values

    redeem = _redemption(bond) + final_coupon
    terminal_conv = kappa * spots + (
        0.0 if terms.conversion_forfeits_accrued else final_coupon
    )
    terminal = np.maximum(terminal_conv, redeem)
    return solve_backward(
        spots,
        times,
        market.volatility,
        terminal,
        coefficients,
        on_node,
        restart,
        rannacher,
    )


def nodal_derivatives(
    spots: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives at every node, three-point non-uniform stencils.

    Interior nodes use the centred stencil; the two boundary nodes copy
    their neighbour's gamma and use a one-sided delta.
    """
    hm = spots[1:-1] - spots[:-2]
    hp = spots[2:] - spots[1:-1]
    vm, v0, vp = values[:-2], values[1:-1], values[2:]
    d_int = (
        (-hp / (hm * (hm + hp))) * vm
        + ((hp - hm) / (hm * hp)) * v0
        + (hm / (hp * (hm + hp))) * vp
    )
    g_int = 2.0 * (vm / (hm * (hm + hp)) - v0 / (hm * hp) + vp / (hp * (hm + hp)))
    delta = np.empty_like(values)
    gamma = np.empty_like(values)
    delta[1:-1], gamma[1:-1] = d_int, g_int
    delta[0] = (values[1] - values[0]) / (spots[1] - spots[0])
    delta[-1] = (values[-1] - values[-2]) / (spots[-1] - spots[-2])
    gamma[0], gamma[-1] = g_int[0], g_int[-1]
    return delta, gamma


@dataclass(frozen=True)
class GridSolution:
    """Values at ``t0`` on the grid plus nodal Greeks, with cubic lookup off-node."""

    spots: np.ndarray
    values: np.ndarray
    spot_index: int

    def __post_init__(self) -> None:
        delta, gamma = nodal_derivatives(self.spots, self.values)
        object.__setattr__(self, "deltas", delta)
        object.__setattr__(self, "gammas", gamma)

    def _cubic(self, series: np.ndarray, spot: float) -> float:
        return float(CubicSpline(self.spots, series)(spot))

    def value_at(self, spot: float) -> float:
        return self._cubic(self.values, spot)

    def delta_at(self, spot: float) -> float:
        return self._cubic(self.deltas, spot)

    def gamma_at(self, spot: float) -> float:
        return self._cubic(self.gammas, spot)

    @property
    def value(self) -> float:
        return float(self.values[self.spot_index])

    @property
    def delta(self) -> float:
        return float(self.deltas[self.spot_index])

    @property
    def gamma(self) -> float:
        return float(self.gammas[self.spot_index])


@dataclass(frozen=True)
class JTDPricingResult:
    """Per-valuation outputs (spec §01). Money amounts are per bond.

    ``*_points`` properties quote in points of par (``100 V / N``).
    ``delta`` is shares per bond; ``delta_pct_parity = Δ/κ``.
    """

    face_value: float
    conversion_ratio: float
    spot: float
    dirty_price: float
    accrued: float
    bond_floor: float | None
    default_value: float
    solution: GridSolution
    floor_solution: GridSolution | None

    @property
    def clean_price(self) -> float:
        return self.dirty_price - self.accrued

    @property
    def parity(self) -> float:
        return self.conversion_ratio * self.spot

    def points(self, amount: float) -> float:
        return 100.0 * amount / self.face_value

    @property
    def dirty_points(self) -> float:
        return self.points(self.dirty_price)

    @property
    def clean_points(self) -> float:
        return self.points(self.clean_price)

    @property
    def conversion_premium(self) -> float:
        """Clean price over parity, minus one."""
        return self.clean_price / self.parity - 1.0 if self.parity > 0 else np.inf

    @property
    def investment_premium(self) -> float | None:
        """Dirty price over the (dirty) bond floor, minus one."""
        if self.bond_floor is None:
            return None
        return self.dirty_price / self.bond_floor - 1.0

    @property
    def delta(self) -> float:
        return self.solution.delta

    @property
    def delta_pct_parity(self) -> float:
        return self.delta / self.conversion_ratio if self.conversion_ratio > 0 else 0.0

    @property
    def gamma(self) -> float:
        return self.solution.gamma

    @property
    def gamma_per_1pct(self) -> float:
        """Change in hedge shares for a 1% spot move: ``Γ · S · 1%``."""
        return self.gamma * self.spot * 0.01


def bond_floor_terms(bond: ConvertibleBond) -> ConvertibleBond:
    """Investment value: the same model with ``κ = 0`` (spec §08 reference note).

    With ``κ = 0`` a soft call's trigger ``h·N/κ`` is unreachable, so only
    hard calls survive; puts are kept.
    """
    return replace(bond, conversion_ratio=0.0)


def price(
    bond: ConvertibleBond,
    market: JTDMarketData,
    terms: JTDTerms = JTDTerms(),
    spec: GridSpec = GridSpec(),
    grid: PricingGrid | None = None,
    with_floor: bool = True,
) -> JTDPricingResult:
    """Price one convertible; pass ``grid`` to reuse a grid across bumps."""
    grid = grid or build_grid(bond, market, spec, terms)
    values = solve_grid(bond, market, grid, terms, spec.rannacher)
    solution = GridSolution(grid.spots, values, grid.spot_index)
    floor_solution = None
    if with_floor:
        floor_values = solve_grid(
            bond_floor_terms(bond), market, grid, terms, spec.rannacher
        )
        floor_solution = GridSolution(grid.spots, floor_values, grid.spot_index)
    acc = accrued_interest(bond)(0.0)
    spot = float(grid.spots[grid.spot_index])
    vd = default_value(bond, market, np.array([spot]), acc)[0]
    if market.recovery_convention is RecoveryConvention.MARKET_VALUE:
        vd = market.recovery * solution.value
    return JTDPricingResult(
        face_value=bond.face_value,
        conversion_ratio=bond.conversion_ratio,
        spot=spot,
        dirty_price=solution.value,
        accrued=acc,
        bond_floor=floor_solution.value if floor_solution else None,
        default_value=float(vd),
        solution=solution,
        floor_solution=floor_solution,
    )
