"""Generic backward solver for the one-factor JTD pricing PDE.

Solves, backward from ``t_K`` to ``t_0``,

    ∂V/∂t + ½σ²S² V_SS + μ(S,t) S V_S − c(S,t) V + f(S,t) = 0

on a (possibly non-uniform) grid in S, where the caller supplies the
drift rate ``μ``, the kill rate ``c`` and the source ``f`` at each step.
For the convertible, ``μ = r − q − b + ηλ``, ``c = r + λ`` and
``f = λ V_D``; CDS legs and European options reuse the same solver
with different ``f`` and terminal values — which is the point of the
spec's "price both CDS legs with the same PDE operator" (§05).

Numerics (spec §06):
  - central differences on the non-uniform stencil; a node switches to
    one-sided upwinding when a central off-diagonal would go negative
    (keeps the matrix an M-matrix, so no spurious oscillations where
    the ηλS drift is large near S = 0);
  - ``S = 0`` row: derivative terms vanish, leaving an ODE in time;
  - ``S = Smax`` row: linearity (V_SS = 0) with a one-sided V_S — *not*
    a Dirichlet ``V = κS``, which would be wrong for κ = 0;
  - Crank–Nicolson, with Rannacher restarts (two fully implicit half
    steps) after maturity and after every flagged event;
  - one tridiagonal solve per (sub)step.

Coefficients are evaluated once per step at the step's mid-time and
held constant across it. The time grid puts hazard pillars on nodes,
so ``λ0(t)`` really is constant within each step.
"""

from collections.abc import Callable

import numpy as np
from scipy.linalg import solve_banded

# t -> (drift rate μ, kill rate c, source f); each shape (M+1,), f may be (M+1, k)
Coefficients = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]
# (time index k, values at t_k) -> values after events/constraints at t_k
NodeHook = Callable[[int, np.ndarray], np.ndarray]


def operator_bands(
    spots: np.ndarray, volatility: float, drift: np.ndarray, kill: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal bands ``(lower, diag, upper)`` of the spatial operator.

    ``(A V)_i = lower_i V_{i-1} + diag_i V_i + upper_i V_{i+1}``;
    ``lower[0]`` and ``upper[-1]`` are unused (zero).
    """
    n = len(spots)
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)

    hm = spots[1:-1] - spots[:-2]
    hp = spots[2:] - spots[1:-1]
    s = spots[1:-1]
    a = 0.5 * volatility**2 * s**2
    b = drift[1:-1] * s
    lo = (2.0 * a - b * hp) / (hm * (hm + hp))
    up = (2.0 * a + b * hm) / (hp * (hm + hp))
    upwind = (lo < 0.0) | (up < 0.0)
    lo = np.where(upwind, 2.0 * a / (hm * (hm + hp)) + np.maximum(-b, 0.0) / hm, lo)
    up = np.where(upwind, 2.0 * a / (hp * (hm + hp)) + np.maximum(b, 0.0) / hp, up)
    lower[1:-1] = lo
    upper[1:-1] = up
    diag[1:-1] = -(lo + up) - kill[1:-1]

    # S = 0: a = b = 0, only the kill term survives.
    diag[0] = -kill[0]
    # S = Smax: V_SS = 0, backward one-sided V_S.
    h = spots[-1] - spots[-2]
    b_top = drift[-1] * spots[-1]
    lower[-1] = -b_top / h
    diag[-1] = b_top / h - kill[-1]
    return lower, diag, upper


def _apply(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, v: np.ndarray
) -> np.ndarray:
    if v.ndim == 2:
        lower, diag, upper = lower[:, None], diag[:, None], upper[:, None]
    out = diag * v
    out[1:] += lower[1:] * v[:-1]
    out[:-1] += upper[:-1] * v[1:]
    return out


def theta_step(
    spots: np.ndarray,
    volatility: float,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    dt: float,
    theta: float,
) -> np.ndarray:
    """One θ-scheme step of size ``dt`` backward in time.

    ``(I − θΔt A) V^k = (I + (1−θ)Δt A) V^{k+1} + Δt f``; θ = ½ is
    Crank–Nicolson, θ = 1 fully implicit.
    """
    drift, kill, source = coefficients
    lower, diag, upper = operator_bands(spots, volatility, drift, kill)
    rhs = values + (1.0 - theta) * dt * _apply(lower, diag, upper, values) + dt * source
    ab = np.empty((3, len(spots)))
    ab[0, 0] = 0.0
    ab[0, 1:] = -theta * dt * upper[:-1]
    ab[1] = 1.0 - theta * dt * diag
    ab[2, :-1] = -theta * dt * lower[1:]
    ab[2, -1] = 0.0
    return solve_banded((1, 1), ab, rhs, check_finite=False)


def solve_backward(
    spots: np.ndarray,
    times: np.ndarray,
    volatility: float,
    terminal: np.ndarray,
    coefficients: Coefficients,
    on_node: NodeHook | None = None,
    restart_indices: frozenset[int] = frozenset(),
    rannacher: bool = True,
) -> np.ndarray:
    """Values at ``times[0]`` from ``terminal`` at ``times[-1]``.

    ``on_node(k, V)`` runs after the solve arrives at every ``t_k`` with
    ``k < K`` (constraints, coupons, dividends). A step *leaving* a node
    in ``restart_indices`` (or leaving maturity) is replaced by two
    fully implicit half steps when ``rannacher`` is on.
    """
    last = len(times) - 1
    values = np.array(terminal, dtype=float, copy=True)
    for k in range(last - 1, -1, -1):
        t0, t1 = times[k], times[k + 1]
        dt = t1 - t0
        if rannacher and (k + 1 == last or (k + 1) in restart_indices):
            half = 0.5 * dt
            values = theta_step(
                spots, volatility, coefficients(t1 - 0.5 * half), values, half, 1.0
            )
            values = theta_step(
                spots, volatility, coefficients(t0 + 0.5 * half), values, half, 1.0
            )
        else:
            values = theta_step(
                spots, volatility, coefficients(0.5 * (t0 + t1)), values, dt, 0.5
            )
        if on_node is not None:
            values = on_node(k, values)
    return values
