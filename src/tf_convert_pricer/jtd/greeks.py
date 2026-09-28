"""Greeks and risk report for the JTD pricer (spec §07).

Two kinds of sensitivity:

  - **Grid Greeks** (delta, gamma) come straight from the ``t0`` solution
    with non-uniform three-point stencils — no extra solves.
  - **Bumped Greeks** re-run the full solve on the *same* space and time
    grid (so grid noise cancels) with central differences.

Credit handling on bumps (spec §07): when the market carries a CDS curve,
bumps to rates, recovery or the CDS curve re-bootstrap ``λ0(t)`` with CDS
spreads held fixed. Without a CDS curve the hazard curve *is* the input:
  - CS01 bumps ``λ0`` by ``1bp / (1 − R)`` (the spec's reference-run convention);
  - the recovery bump rescales ``λ0`` by ``(1 − R)/(1 − R')`` so the
    ``λ(1 − R)`` spread proxy stays fixed — an approximation of
    "spreads fixed" that ignores the ``p > 0`` convexity correction;
  - the rate bump leaves ``λ0`` alone.
Vol, ``p``, borrow and dividend bumps never re-bootstrap (the spec lists
re-bootstrapping only for rates, recovery and credit).

Units: price-like sensitivities are in **points of par** (``100 ΔV / N``)
per stated bump; delta is shares per bond; gamma shares per bond per
unit of stock price.

Sticky assumption: a spot move holds ``λ0(t)`` fixed, so the hazard moves
along ``λ(S)``. Delta therefore *includes* the credit channel; the
``credit_delta`` field shows how much. Don't hedge that credit move again
with CDS.
"""

from dataclasses import dataclass, replace

import numpy as np

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.jtd.calibration import bootstrap_hazard
from tf_convert_pricer.jtd.grid import GridSpec
from tf_convert_pricer.jtd.market import HazardCurve, JTDMarketData
from tf_convert_pricer.jtd.pricer import (
    GridSolution,
    JTDPricingResult,
    JTDTerms,
    PricingGrid,
    accrued_interest,
    build_grid,
    coupon_schedule,
    price,
    solve_grid,
)

ONE_BP = 1e-4
ONE_DAY = 1.0 / 365.0


class _Engine:
    """Reprices one bond on one fixed grid; knows how to re-bootstrap."""

    def __init__(
        self, bond: ConvertibleBond, terms: JTDTerms, spec: GridSpec, grid: PricingGrid
    ) -> None:
        self.bond, self.terms, self.spec, self.grid = bond, terms, spec, grid

    def solve(
        self,
        market: JTDMarketData,
        bond: ConvertibleBond | None = None,
        grid: PricingGrid | None = None,
    ) -> GridSolution:
        grid = grid or self.grid
        values = solve_grid(
            bond or self.bond, market, grid, self.terms, self.spec.rannacher
        )
        return GridSolution(grid.spots, values, grid.spot_index)

    def value(self, market: JTDMarketData) -> float:
        return self.solve(market).value

    def recalibrated(self, market: JTDMarketData) -> JTDMarketData:
        """Re-bootstrap ``hazard`` from ``market.cds_curve`` if there is one."""
        if market.cds_curve is None:
            return market
        hazard = bootstrap_hazard(
            market, market.cds_curve, self.spec, initial=market.hazard
        )
        return replace(market, hazard=hazard)

    def central(self, up: JTDMarketData, down: JTDMarketData) -> float:
        """``(V(up) − V(down)) / 2`` in points, re-bootstrapping each side."""
        diff = self.value(self.recalibrated(up)) - self.value(self.recalibrated(down))
        return 100.0 * diff / self.bond.face_value / 2.0


def frozen_hazard_market(market: JTDMarketData, spot: float) -> JTDMarketData:
    """``p = 0`` with ``λ0(t) ← λ(spot, t)``: the hazard no longer reacts to S."""
    frozen = tuple(
        float(market.intensity(np.array([spot]), t - 1e-9)[0])
        for t in market.hazard.pillars
    )
    curve = HazardCurve(market.hazard.pillars, frozen, spot)
    return replace(market, hazard=curve, hazard_elasticity=0.0, cds_curve=None)


def equity_delta(engine: _Engine, market: JTDMarketData, spot: float) -> float:
    """``Δ_eq``: delta with the hazard frozen at its value at ``spot``."""
    return engine.solve(frozen_hazard_market(market, spot)).delta_at(spot)


def _cs01(engine: _Engine, market: JTDMarketData, pillar: int | None = None) -> float:
    if market.cds_curve is not None:
        up = replace(market, cds_curve=market.cds_curve.shifted(ONE_BP, pillar))
        dn = replace(market, cds_curve=market.cds_curve.shifted(-ONE_BP, pillar))
        return engine.central(up, dn)
    shift = ONE_BP / (1.0 - market.recovery)
    return engine.central(
        replace(market, hazard=market.hazard.shifted(shift)),
        replace(market, hazard=market.hazard.shifted(-shift)),
    )


def _recovery_bumped(market: JTDMarketData, recovery: float) -> JTDMarketData:
    bumped = replace(market, recovery=recovery)
    if market.cds_curve is None:
        factor = (1.0 - market.recovery) / (1.0 - recovery)
        bumped = replace(bumped, hazard=market.hazard.scaled(factor))
    return bumped


def rolled_bond(bond: ConvertibleBond, dt: float) -> tuple[ConvertibleBond, float]:
    """Bond seen from ``t0 + dt`` and the coupon cash paid in ``(0, dt]``.

    Call periods already started stay active (key clipped to 0); put
    dates inside the window are dropped (assumed not exercised).
    """
    paid = sum(a for t, a in coupon_schedule(bond) if t <= dt + 1e-12)
    calls = None
    if bond.call_schedule:
        shifted: dict[float, float] = {}
        for t, level in sorted(bond.call_schedule.items()):
            shifted[max(t - dt, 0.0)] = level
        calls = shifted
    puts = None
    if bond.put_schedule:
        puts = {
            t - dt: level for t, level in bond.put_schedule.items() if t - dt > -1e-12
        } or None
    rolled = replace(
        bond, maturity=bond.maturity - dt, call_schedule=calls, put_schedule=puts
    )
    return rolled, paid


def _rolled_market(market: JTDMarketData, dt: float) -> tuple[JTDMarketData, float]:
    """Market inputs unchanged except discrete dividends move ``dt`` closer."""
    paid = sum(d for t, d in market.discrete_dividends if t <= dt + 1e-12)
    divs = tuple((t - dt, d) for t, d in market.discrete_dividends if t > dt + 1e-12)
    return replace(market, discrete_dividends=divs), paid


@dataclass(frozen=True)
class CarryBreakdown:
    """Daily carry of a delta-hedged long convert, in points of par.

    ``coupon_accrual − Δ × dividends + short_rebate``, with the rebate on
    the short stock proceeds at ``r − b``. Bond financing is not included.
    """

    coupon_accrual: float
    dividends_owed: float
    short_rebate: float

    @property
    def total(self) -> float:
        return self.coupon_accrual - self.dividends_owed + self.short_rebate


@dataclass(frozen=True)
class JTDRiskReport:
    """Every Greek of spec §07 for one valuation. See module docstring for units."""

    pricing: JTDPricingResult
    delta: float
    delta_pct_parity: float
    gamma: float
    gamma_per_1pct: float
    equity_delta: float
    credit_delta: float
    vega: float
    cs01: float
    cs01_buckets: tuple[tuple[float, float], ...]
    rho: float
    recovery_sensitivity: float
    p_price_sensitivity: float
    p_delta_sensitivity: float
    borrow_sensitivity: float
    dividend_sensitivity: float
    theta: float
    carry: CarryBreakdown
    jtd_naked: float
    jtd_hedged: float


def risk_report(
    bond: ConvertibleBond,
    market: JTDMarketData,
    terms: JTDTerms = JTDTerms(),
    spec: GridSpec = GridSpec(),
    cds_hedge_notional: float = 0.0,
    theta_days: float = 1.0,
) -> JTDRiskReport:
    """Full risk report per spec §07.

    ``cds_hedge_notional`` is the CDS protection held per bond, used only
    in the hedged jump-to-default. ``theta_days`` is the roll in calendar
    days (ACT/365); pass 3 to roll a Friday to Monday.
    """
    grid = build_grid(bond, market, spec, terms)
    engine = _Engine(bond, terms, spec, grid)
    base = price(bond, market, terms, spec, grid)
    pts = 100.0 / bond.face_value
    spot = base.spot

    delta_eq = equity_delta(engine, market, spot)

    vol_up = replace(market, volatility=market.volatility + 0.01)
    vol_dn = replace(market, volatility=market.volatility - 0.01)
    vega = pts * (engine.value(vol_up) - engine.value(vol_dn)) / 2.0

    cs01 = _cs01(engine, market)
    buckets: tuple[tuple[float, float], ...] = ()
    if market.cds_curve is not None:
        buckets = tuple(
            (t, _cs01(engine, market, j))
            for j, t in enumerate(market.cds_curve.pillars)
        )

    rho = engine.central(
        replace(market, risk_free_rate=market.risk_free_rate + ONE_BP),
        replace(market, risk_free_rate=market.risk_free_rate - ONE_BP),
    )
    recovery = engine.central(
        _recovery_bumped(market, market.recovery + 0.05),
        _recovery_bumped(market, market.recovery - 0.05),
    )

    p_up = engine.solve(
        replace(market, hazard_elasticity=market.hazard_elasticity + 0.25)
    )
    p_dn = engine.solve(
        replace(market, hazard_elasticity=max(market.hazard_elasticity - 0.25, 0.0))
    )
    p_width = (
        market.hazard_elasticity + 0.25 - max(market.hazard_elasticity - 0.25, 0.0)
    )
    p_price = pts * (p_up.value - p_dn.value) / p_width * 0.25
    p_delta = (p_up.delta - p_dn.delta) / p_width * 0.25

    borrow = (
        pts
        * (
            engine.value(replace(market, borrow_cost=market.borrow_cost + 10 * ONE_BP))
            - engine.value(
                replace(market, borrow_cost=market.borrow_cost - 10 * ONE_BP)
            )
        )
        / 2.0
    )
    dividend = (
        pts
        * (
            engine.value(
                replace(market, dividend_yield=market.dividend_yield + 10 * ONE_BP)
            )
            - engine.value(
                replace(market, dividend_yield=market.dividend_yield - 10 * ONE_BP)
            )
        )
        / 2.0
    )

    dt = theta_days * ONE_DAY
    later_bond, coupon_paid = rolled_bond(bond, dt)
    later_market, divs_paid = _rolled_market(market, dt)
    later_value = engine.solve(later_market, later_bond, grid.rolled(dt)).value
    theta = pts * (later_value + coupon_paid - base.dirty_price)

    coupon_accrual = accrued_interest(later_bond)(0.0) + coupon_paid - base.accrued
    carry = CarryBreakdown(
        coupon_accrual=pts * coupon_accrual,
        dividends_owed=pts
        * base.delta
        * (spot * market.dividend_yield * dt + divs_paid),
        short_rebate=pts
        * base.delta
        * spot
        * (market.risk_free_rate - market.borrow_cost)
        * dt,
    )

    jtd_naked = pts * (base.default_value - base.dirty_price)
    jtd_hedged = pts * (
        base.delta * market.jump_size * spot
        - (base.dirty_price - base.default_value)
        + cds_hedge_notional * (1.0 - market.recovery)
    )

    return JTDRiskReport(
        pricing=base,
        delta=base.delta,
        delta_pct_parity=base.delta_pct_parity,
        gamma=base.gamma,
        gamma_per_1pct=base.gamma_per_1pct,
        equity_delta=delta_eq,
        credit_delta=base.delta - delta_eq,
        vega=vega,
        cs01=cs01,
        cs01_buckets=buckets,
        rho=rho,
        recovery_sensitivity=recovery,
        p_price_sensitivity=p_price,
        p_delta_sensitivity=p_delta,
        borrow_sensitivity=borrow,
        dividend_sensitivity=dividend,
        theta=theta,
        carry=carry,
        jtd_naked=jtd_naked,
        jtd_hedged=jtd_hedged,
    )


@dataclass(frozen=True)
class LadderRow:
    spot: float
    price_points: float
    delta_pct_parity: float
    equity_delta_pct_parity: float
    floor_points: float


def spot_ladder(
    bond: ConvertibleBond,
    market: JTDMarketData,
    spots: list[float],
    terms: JTDTerms = JTDTerms(),
    spec: GridSpec = GridSpec(),
) -> list[LadderRow]:
    """Price / delta / Δ_eq / floor across spots (spec §08 table).

    One base solve covers every spot via cubic interpolation, with
    ``λ0`` held fixed (sticky hazard). ``Δ_eq`` needs one frozen-hazard
    solve per spot.
    """
    grid = build_grid(bond, market, spec, terms)
    engine = _Engine(bond, terms, spec, grid)
    base = price(bond, market, terms, spec, grid)
    kappa = bond.conversion_ratio
    pts = 100.0 / bond.face_value
    rows = []
    for s in spots:
        rows.append(
            LadderRow(
                spot=s,
                price_points=pts * base.solution.value_at(s),
                delta_pct_parity=base.solution.delta_at(s) / kappa,
                equity_delta_pct_parity=equity_delta(engine, market, s) / kappa,
                floor_points=pts * base.floor_solution.value_at(s),
            )
        )
    return rows
