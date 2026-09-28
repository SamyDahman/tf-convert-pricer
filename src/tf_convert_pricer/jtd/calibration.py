"""Calibration for the JTD model (spec §05).

- ``bootstrap_hazard``: piecewise-constant ``λ0(t)`` from CDS par
  spreads, pillar by pillar, pricing both CDS legs with the *same*
  PDE operator as the bond (so a stock-dependent hazard, ``p > 0``,
  is priced consistently — the shortcut ``λ0 = s/(1−R)`` overstates
  long-dated spreads then, and is only the root-finder's first guess).
- ``convert_implied_volatility``: the σ that reprices a market convert.
- ``price_european_option`` / ``option_implied_volatility``: listed
  options priced in this same model, for calibrating σ without
  double-counting default (never plug a BS implied vol in directly).
  American listed options are out of scope for v1.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, newton

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.jtd.analytic import black_scholes_call
from tf_convert_pricer.jtd.grid import GridSpec, smax_for, space_grid, time_grid
from tf_convert_pricer.jtd.market import CDSCurve, HazardCurve, JTDMarketData
from tf_convert_pricer.jtd.pde import solve_backward
from tf_convert_pricer.jtd.pricer import JTDTerms, PricingGrid, build_grid, solve_grid


def _drift(market: JTDMarketData, lam: np.ndarray) -> np.ndarray:
    return (
        market.risk_free_rate
        - market.dividend_yield
        - market.borrow_cost
        + market.jump_size * lam
    )


def model_par_spread(
    market: JTDMarketData, maturity: float, spec: GridSpec = GridSpec()
) -> float:
    """``U(S0, 0) / W(S0, 0)`` for a CDS of the given maturity under ``market.hazard``.

    Protection ``U`` has source ``λ(1 − R)``; the (continuous) premium
    annuity ``W`` has source ``1``. Both have zero terminal value and
    are solved together as two right-hand sides of one PDE.
    """
    spot = market.spot
    smax = smax_for(spot, spot, market.volatility, maturity, spec)
    spots, j0 = space_grid(spot, smax, spec)
    times = time_grid(maturity, list(market.hazard.pillars), spec.steps_per_year)
    loss = 1.0 - market.recovery

    def coefficients(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lam = market.intensity(spots, t)
        source = np.column_stack((lam * loss, np.ones_like(spots)))
        return _drift(market, lam), market.risk_free_rate + lam, source

    legs = solve_backward(
        spots,
        times,
        market.volatility,
        np.zeros((len(spots), 2)),
        coefficients,
        rannacher=False,
    )
    protection, annuity = legs[j0]
    return float(protection / annuity)


def bootstrap_hazard(
    market: JTDMarketData,
    cds: CDSCurve,
    spec: GridSpec = GridSpec(),
    initial: HazardCurve | None = None,
) -> HazardCurve:
    """Sequential bootstrap: for pillar ``j`` solve for ``λ0`` on ``(T_{j−1}, T_j]``.

    Uses ``market`` for everything except ``hazard`` (σ, r, q, b, p, η,
    λmax, R). The resulting curve's reference spot is ``market.spot``.
    ``initial`` warm-starts the secant iteration (bumped re-bootstraps).
    """
    solved: list[float] = []
    for j, (pillar, spread) in enumerate(zip(cds.pillars, cds.spreads)):
        pillars = cds.pillars[: j + 1]

        def mismatch(
            level: float,
            pillars: tuple[float, ...] = pillars,
            pillar: float = pillar,
            spread: float = spread,
        ) -> float:
            curve = HazardCurve(pillars, (*solved, max(level, 0.0)), market.spot)
            return (
                model_par_spread(replace(market, hazard=curve), pillar, spec) - spread
            )

        if initial is not None and j < len(initial.intensities):
            guess = initial.intensities[j]
        else:
            guess = spread / (1.0 - market.recovery)
        try:
            level = newton(
                mismatch, guess, x1=guess * 1.01 + 1e-5, tol=1e-12, maxiter=30
            )
            if level < 0 or abs(mismatch(level)) > 1e-9:
                raise RuntimeError
        except (RuntimeError, OverflowError):
            level = brentq(mismatch, 0.0, market.lambda_max, xtol=1e-12)
        solved.append(float(level))
    return HazardCurve(tuple(cds.pillars), tuple(solved), market.spot)


def calibrate_market(
    market: JTDMarketData, cds: CDSCurve, spec: GridSpec = GridSpec()
) -> JTDMarketData:
    """``market`` with ``hazard`` bootstrapped from ``cds`` and ``cds_curve`` recorded."""
    initial = market.hazard if market.cds_curve is not None else None
    hazard = bootstrap_hazard(market, cds, spec, initial)
    return replace(market, hazard=hazard, cds_curve=cds)


def convert_implied_volatility(
    bond: ConvertibleBond,
    market: JTDMarketData,
    target_dirty_price: float,
    terms: JTDTerms = JTDTerms(),
    spec: GridSpec = GridSpec(),
    bounds: tuple[float, float] = (0.01, 2.0),
) -> float:
    """σ such that the model dirty price equals ``target_dirty_price`` (mode a).

    The grid is fixed at the upper vol bound so that ``Smax`` is wide
    enough for every trial σ and the root-finder sees a smooth function.
    """
    grid = build_grid(bond, replace(market, volatility=bounds[1]), spec, terms)

    def mismatch(vol: float) -> float:
        bumped = replace(market, volatility=vol)
        return (
            float(
                solve_grid(bond, bumped, grid, terms, spec.rannacher)[grid.spot_index]
            )
            - target_dirty_price
        )

    return brentq(mismatch, *bounds, xtol=1e-10)


def price_european_option(
    market: JTDMarketData,
    strike: float,
    maturity: float,
    is_call: bool,
    spec: GridSpec = GridSpec(),
    grid: PricingGrid | None = None,
) -> float:
    """European option on the stock in the JTD model.

    On default the stock drops to ``(1 − η)S`` and (assumption) diffuses
    on as plain Black–Scholes with no further default, so the option is
    then worth a BS price on the post-default stock with the remaining
    time. For ``η = 1`` that is ``0`` for calls and ``K e^{−r(T−t)}`` for
    puts. (The spec writes ``V_D = K`` for puts, which is the *American*
    value; for a European the strike is only received at ``T``.)
    """
    if grid is None:
        smax = smax_for(market.spot, strike, market.volatility, maturity, spec)
        spots, j0 = space_grid(market.spot, smax, spec, (strike,))
        grid = PricingGrid(
            spots,
            j0,
            time_grid(maturity, list(market.hazard.pillars), spec.steps_per_year),
        )
    spots = grid.spots
    sign = 1.0 if is_call else -1.0
    payoff = np.maximum(sign * (spots - strike), 0.0)
    r, yld = market.risk_free_rate, market.dividend_yield + market.borrow_cost
    survivor = (1.0 - market.jump_size) * spots

    def post_default(t: float) -> np.ndarray:
        tau = maturity - t
        call = np.asarray(
            black_scholes_call(survivor, strike, tau, r, market.volatility, yld)
        )
        if is_call:
            return call
        return call - survivor * np.exp(-yld * tau) + strike * np.exp(-r * tau)

    def coefficients(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lam = market.intensity(spots, t)
        return _drift(market, lam), r + lam, lam * post_default(t)

    values = solve_backward(
        spots,
        grid.times,
        market.volatility,
        payoff,
        coefficients,
        rannacher=spec.rannacher,
    )
    return float(values[grid.spot_index])


def option_implied_volatility(
    market: JTDMarketData,
    strike: float,
    maturity: float,
    is_call: bool,
    target_price: float,
    spec: GridSpec = GridSpec(),
    bounds: tuple[float, float] = (0.01, 2.0),
) -> float:
    """Diffusion σ that reprices a listed option in the JTD model (mode b)."""
    smax = smax_for(market.spot, strike, bounds[1], maturity, spec)
    spots, j0 = space_grid(market.spot, smax, spec, (strike,))
    grid = PricingGrid(
        spots, j0, time_grid(maturity, list(market.hazard.pillars), spec.steps_per_year)
    )

    def mismatch(vol: float) -> float:
        bumped = replace(market, volatility=vol)
        return (
            price_european_option(bumped, strike, maturity, is_call, spec, grid)
            - target_price
        )

    return brentq(mismatch, *bounds, xtol=1e-10)
