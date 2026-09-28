"""Space and time grids for the JTD PDE (spec §06).

Space: ``S ∈ [0, Smax]`` with ``Smax ≥ 4 max(S0, N/κ) e^{3σ√T}``.
Either uniform or sinh-stretched around ``S0``; in both cases ``S0``
sits exactly on a node so the price and grid Greeks are read off
without interpolation.

Time: every event date (coupon, dividend, call start, put date, hazard
pillar) is an exact node; the gaps between events are split into equal
steps no larger than ``1 / steps_per_year``.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import asinh, ceil, exp, floor, sqrt

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Numerical resolution knobs.

    ``concentration`` is the sinh stretching width as a fraction of
    ``S0`` — smaller means more nodes packed near spot. ``None`` gives a
    uniform grid (what the spec's reference prototype used).

    The defaults are chosen to meet the spec's convergence target
    (halving ΔS and Δt moves price < 0.02 pts, delta < 0.001) on the
    reference convertible at interactive speed; see
    ``tests/test_jtd_pricer.py::TestConvergence``.
    """

    space_nodes: int = 400
    steps_per_year: int = 400
    concentration: float | None = 0.3
    smax_multiplier: float = 4.0
    smax: float | None = None
    rannacher: bool = True

    def refined(self) -> "GridSpec":
        """Halve both ΔS and Δt — used by the convergence check."""
        return GridSpec(
            space_nodes=2 * self.space_nodes,
            steps_per_year=2 * self.steps_per_year,
            concentration=self.concentration,
            smax_multiplier=self.smax_multiplier,
            smax=self.smax,
            rannacher=self.rannacher,
        )


def smax_for(
    spot: float, strike: float, volatility: float, maturity: float, spec: GridSpec
) -> float:
    """Upper boundary per spec §06, or the explicit override."""
    if spec.smax is not None:
        return spec.smax
    return (
        spec.smax_multiplier
        * max(spot, strike)
        * exp(3.0 * volatility * sqrt(maturity))
    )


def space_grid(
    spot: float, smax: float, spec: GridSpec, pins: tuple[float, ...] = ()
) -> tuple[np.ndarray, int]:
    """Nodes ``S_0 = 0 < ... < S_M ≈ smax`` and the index of ``spot``.

    ``pins`` (e.g. conversion price, soft-call trigger) are moved onto
    the nearest free node, so a constraint that switches on at that
    level switches on exactly there rather than at the next node up.

    Uniform: ``ΔS`` is shrunk slightly so that ``spot / ΔS`` is an
    integer. Sinh: ``S(ξ) = S0 + α sinh(c1 + (c2 - c1) ξ)`` on uniform
    ``ξ``; ``c2`` (hence ``Smax``) is nudged *up* so that ``S0`` lands
    exactly on node ``j0``.
    """
    m = spec.space_nodes
    if spot <= 0 or smax <= spot:
        raise ValueError("need 0 < spot < smax")
    if spec.concentration is None:
        j0 = max(1, round(spot / (smax / m)))
        ds = spot / j0
        n = ceil(smax / ds)
        return _pin(ds * np.arange(n + 1), j0, pins), j0
    alpha = spec.concentration * spot
    c1 = asinh(-spot / alpha)
    c2 = asinh((smax - spot) / alpha)
    j0 = max(1, floor(m * (-c1) / (c2 - c1)))
    c2 = c1 * (1.0 - m / j0)
    xi = np.arange(m + 1) / m
    nodes = spot + alpha * np.sinh(c1 + (c2 - c1) * xi)
    nodes[0] = 0.0
    nodes[j0] = spot
    return _pin(nodes, j0, pins), j0


def _pin(nodes: np.ndarray, spot_index: int, pins: tuple[float, ...]) -> np.ndarray:
    """Move the nearest interior node (other than spot's) onto each pin."""
    nodes = nodes.copy()
    for level in pins:
        if not nodes[1] < level < nodes[-2]:
            continue
        i = int(np.argmin(np.abs(nodes - level)))
        if (
            i != spot_index
            and 0 < i < len(nodes) - 1
            and nodes[i - 1] < level < nodes[i + 1]
        ):
            nodes[i] = level
    return nodes


def time_grid(
    maturity: float, event_times: list[float], steps_per_year: int
) -> np.ndarray:
    """Times ``0 = t_0 < ... < t_K = maturity`` containing every event exactly."""
    knots = sorted(
        {0.0, maturity, *(t for t in event_times if 1e-9 < t < maturity - 1e-9)}
    )
    merged: list[float] = [knots[0]]
    for t in knots[1:]:
        if t - merged[-1] > 1e-9:
            merged.append(t)
    times: list[float] = [0.0]
    for a, b in pairwise(merged):
        n = max(1, ceil((b - a) * steps_per_year - 1e-9))
        times.extend(a + (b - a) * np.arange(1, n + 1) / n)
    return np.asarray(times)


def node_index(times: np.ndarray, t: float) -> int | None:
    """Index of the time node equal to ``t`` (within 1e-9), else ``None``."""
    k = int(np.searchsorted(times, t - 1e-9))
    if k < len(times) and abs(times[k] - t) < 1e-9:
        return k
    return None
