"""Path simulation and per-frame series for the JTD model, plus the
spot profiles (value / delta vs. stock) for both models.

Why this looks different from ``simulation.py`` (TF): one PDE solve gives
the convertible's value on the whole stock grid, and delta/gamma come off
that grid with smooth stencils. So a whole path needs only a handful of
solves — base, σ ± 1%, λ0 ± 1bp/(1−R), a one-day roll — each read at every
frame's spot by cubic interpolation. No per-frame bump-and-reprice, and no
smoothing needed.

As in the TF demo, the bond's remaining maturity is held fixed along the
path ("same bond, spot moves"), so each frame's point lies exactly on the
static value-vs-stock curve.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from tf_convert_pricer.greeks import delta as tf_delta
from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.jtd.greeks import (
    ONE_BP,
    ONE_DAY,
    frozen_hazard_market,
    rolled_bond,
)
from tf_convert_pricer.jtd.grid import GridSpec
from tf_convert_pricer.jtd.market import JTDMarketData
from tf_convert_pricer.jtd.pricer import (
    GridSolution,
    JTDTerms,
    PricingGrid,
    build_grid,
    default_value,
    solve_grid,
)
from tf_convert_pricer.market import MarketData
from tf_convert_pricer.pricer import price as tf_price

# Coarser grid for the many frozen-hazard solves behind Δ_eq.
EQUITY_DELTA_GRID = GridSpec(space_nodes=250, steps_per_year=150)


@dataclass(frozen=True)
class SimulatedPath:
    spots: np.ndarray
    default_frame: int | None  # first frame at which the issuer has defaulted


def simulate_jtd_path(
    market: JTDMarketData, n_days: int, n_frames: int, seed: int, allow_default: bool = True, substeps: int = 4
) -> SimulatedPath:
    """Pre-default JTD dynamics with a default event (spec §02).

    ``dS/S = (r − q − b + ηλ(S))dt + σ dW`` until default, which arrives
    with intensity ``λ(S)``; then the stock drops to ``(1 − η)S`` and
    stays there. Log-Euler with ``substeps`` per frame.
    """
    rng = np.random.default_rng(seed)
    dt = n_days / 252.0 / n_frames / substeps
    carry = market.risk_free_rate - market.dividend_yield - market.borrow_cost
    spots = np.empty(n_frames + 1)
    spots[0] = market.spot
    s = market.spot
    default_frame = None
    for frame in range(1, n_frames + 1):
        for _ in range(substeps):
            if default_frame is not None:
                break
            lam = float(market.intensity(np.array([s]), 0.0)[0])
            if allow_default and rng.random() < 1.0 - np.exp(-lam * dt):
                s *= 1.0 - market.jump_size
                default_frame = frame
                break
            drift = carry + market.jump_size * lam - 0.5 * market.volatility**2
            s *= np.exp(drift * dt + market.volatility * np.sqrt(dt) * rng.standard_normal())
        spots[frame] = s
    return SimulatedPath(spots, default_frame)


@dataclass(frozen=True)
class JTDCurves:
    """Everything needed to read price and Greeks at any spot."""

    grid: PricingGrid
    base: GridSolution
    floor: GridSolution
    vega: np.ndarray  # $ per 1 vol point, per node
    cs01: np.ndarray  # $ per 1bp of spread, per node
    theta: np.ndarray  # $ per calendar day, per node
    equity_delta: CubicSpline  # shares per bond, vs. spot


def solve_curves(
    bond: ConvertibleBond,
    market: JTDMarketData,
    terms: JTDTerms,
    spec: GridSpec = GridSpec(),
    equity_delta_spots: np.ndarray | None = None,
) -> JTDCurves:
    """The handful of PDE solves behind the profile and the time-lapse.

    Bumps follow ``jtd.greeks`` conventions (same grid; CS01 bumps λ0 by
    1bp/(1 − R)), but are kept as whole curves rather than values at S0.
    """
    grid = build_grid(bond, market, spec, terms)

    def solve(mkt: JTDMarketData, b: ConvertibleBond = bond, g: PricingGrid = grid) -> np.ndarray:
        return solve_grid(b, mkt, g, terms, spec.rannacher)

    base = solve(market)
    floor = solve(market, replace(bond, conversion_ratio=0.0))
    vega = (solve(replace(market, volatility=market.volatility + 0.01)) - solve(replace(market, volatility=market.volatility - 0.01))) / 2.0
    shift = ONE_BP / (1.0 - market.recovery)
    cs01 = (solve(replace(market, hazard=market.hazard.shifted(shift))) - solve(replace(market, hazard=market.hazard.shifted(-shift)))) / 2.0
    later, coupon_paid = rolled_bond(bond, ONE_DAY)
    theta = solve(market, later, grid.rolled(ONE_DAY)) + coupon_paid - base

    if equity_delta_spots is None:
        equity_delta_spots = np.linspace(0.1, 2.5, 25) * bond.face_value / max(bond.conversion_ratio, 1e-12)
    eq_grid = build_grid(bond, market, EQUITY_DELTA_GRID, terms)
    eq = [
        GridSolution(eq_grid.spots, solve_grid(bond, frozen_hazard_market(market, s), eq_grid, terms), eq_grid.spot_index).delta_at(s)
        for s in equity_delta_spots
    ]
    return JTDCurves(
        grid=grid,
        base=GridSolution(grid.spots, base, grid.spot_index),
        floor=GridSolution(grid.spots, floor, grid.spot_index),
        vega=vega,
        cs01=cs01,
        theta=theta,
        equity_delta=CubicSpline(equity_delta_spots, eq),
    )


def _at(curves: JTDCurves, series: np.ndarray, spot: float) -> float:
    return float(CubicSpline(curves.grid.spots, series)(spot))


def compute_jtd_series(
    bond: ConvertibleBond,
    market: JTDMarketData,
    curves: JTDCurves,
    path: SimulatedPath,
    n_days: int,
) -> pd.DataFrame:
    """Per-frame price and Greeks, in the TF demo's per-share conventions.

    ``delta``/``equity_delta``/``gamma`` are divided by κ (per share of
    conversion); everything else is $ per bond. After default the bond is
    worth its recovery, and every sensitivity is zero.
    """
    kappa = bond.conversion_ratio
    days = np.linspace(0.0, float(n_days), len(path.spots))
    recovery_cash = market.recovery * bond.face_value
    records = []
    for frame, (day, spot) in enumerate(zip(days, path.spots, strict=True)):
        defaulted = path.default_frame is not None and frame >= path.default_frame
        if defaulted:
            records.append(
                dict(
                    frame=frame, day=day, spot=float(spot), fair_value=recovery_cash, parity=kappa * float(spot),
                    bond_floor=recovery_cash, delta=0.0, equity_delta=0.0, gamma=0.0, vega=0.0,
                    credit_spread_sensitivity=0.0, theta=0.0, hazard=np.nan, jtd_naked=0.0, jtd_hedged=0.0,
                    defaulted=True,
                )
            )
            continue
        value = curves.base.value_at(spot)
        delta = curves.base.delta_at(spot)
        v_d = float(default_value(bond, market, np.array([spot]), 0.0)[0])
        records.append(
            dict(
                frame=frame,
                day=day,
                spot=float(spot),
                fair_value=value,
                parity=kappa * float(spot),
                bond_floor=curves.floor.value_at(spot),
                delta=delta / kappa,
                equity_delta=float(curves.equity_delta(spot)) / kappa,
                gamma=curves.base.gamma_at(spot) / kappa,
                vega=_at(curves, curves.vega, spot),
                credit_spread_sensitivity=_at(curves, curves.cs01, spot),
                theta=_at(curves, curves.theta, spot),
                hazard=float(market.intensity(np.array([spot]), 0.0)[0]),
                jtd_naked=v_d - value,
                jtd_hedged=delta * market.jump_size * float(spot) - (value - v_d),
                defaulted=False,
            )
        )
    return pd.DataFrame(records)


def jtd_profile(bond: ConvertibleBond, market: JTDMarketData, curves: JTDCurves, spots: np.ndarray) -> pd.DataFrame:
    """Value, floor, parity and delta split across ``spots`` (spec §08 figures)."""
    kappa = bond.conversion_ratio
    return pd.DataFrame(
        dict(
            spot=spots,
            value=[curves.base.value_at(s) for s in spots],
            floor=[curves.floor.value_at(s) for s in spots],
            parity=kappa * spots,
            delta=[curves.base.delta_at(s) / kappa for s in spots],
            equity_delta=curves.equity_delta(spots) / kappa,
            hazard=market.intensity(spots, 0.0),
        )
    )


def tf_profile(bond: ConvertibleBond, market: MarketData, spots: np.ndarray, steps: int = 400) -> pd.DataFrame:
    """TF value, floor and delta across ``spots`` (one tree per spot)."""
    floor_bond = replace(bond, conversion_ratio=0.0)
    kappa = bond.conversion_ratio
    rows = []
    for s in spots:
        moved = replace(market, spot=float(s))
        rows.append(
            dict(
                spot=float(s),
                value=tf_price(bond, moved, steps).price,
                floor=tf_price(floor_bond, moved, steps).price,
                delta=tf_delta(bond, moved, steps, bump=0.02 * float(s)) / kappa,
            )
        )
    return pd.DataFrame(rows)
