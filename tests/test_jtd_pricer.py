"""Tests for the JTD convertible pricer.

Organised around the spec's §08 acceptance criteria (T1–T5, reference
run, convergence), plus cross-checks against the TF tree pricer where
the two models must agree, and one test per event mechanic.
"""

from dataclasses import replace
from math import exp

import numpy as np
import pytest

from tf_convert_pricer.examples import busted, equity_like, hybrid
from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.jtd.analytic import black_scholes_call, risky_zero_coupon
from tf_convert_pricer.jtd.examples import (
    PROTOTYPE_GRID,
    common_market,
    reference_convertible,
)
from tf_convert_pricer.jtd.grid import GridSpec
from tf_convert_pricer.jtd.market import HazardCurve, JTDMarketData, RecoveryConvention
from tf_convert_pricer.jtd.pricer import (
    JTDTerms,
    accrued_interest,
    build_grid,
    price,
    solve_grid,
)
from tf_convert_pricer.pricer import price as tf_price


def spec_bond(**overrides) -> ConvertibleBond:
    """N = 100, κ = 1, T = 5, zero coupon, no call/put — override as needed."""
    defaults = dict(
        face_value=100.0,
        coupon_rate=0.0,
        coupon_frequency=2,
        maturity=5.0,
        conversion_ratio=1.0,
    )
    defaults.update(overrides)
    return ConvertibleBond(**defaults)


def four_decimals(model: float, exact: float) -> bool:
    return round(model, 4) == round(exact, 4)


class TestSpecClosedForms:
    """T1–T3 on the prototype grid, "reproduced to 4 decimals"."""

    def test_t1_no_credit(self) -> None:
        # Early conversion never optimal without dividends → European.
        result = price(
            spec_bond(), common_market(), spec=PROTOTYPE_GRID, with_floor=False
        )
        exact = 100 * exp(-0.2) + black_scholes_call(100.0, 100.0, 5.0, 0.04, 0.25)
        assert four_decimals(result.dirty_price, exact)
        assert four_decimals(exact, 112.1857)

    def test_t2_risky_zero(self) -> None:
        market = common_market(intensity=0.03, hazard_elasticity=0.0)
        result = price(
            spec_bond(conversion_ratio=0.0),
            market,
            spec=PROTOTYPE_GRID,
            with_floor=False,
        )
        assert four_decimals(
            result.dirty_price, risky_zero_coupon(100.0, 5.0, 0.04, 0.03, 0.4)
        )

    # T3 (call under JTD) lives in test_jtd_calibration.py next to the option pricer.


class TestLimits:
    """T4: deep ITM → parity; κ = 0 → the straight risky bond."""

    def test_deep_in_the_money(self) -> None:
        market = replace(common_market(0.03, 1.0), spot=300.0)
        result = price(spec_bond(), market, with_floor=False)
        assert result.dirty_price == pytest.approx(300.0, rel=0.01)
        assert result.delta == pytest.approx(1.0, abs=0.01)

    def test_zero_conversion_ratio_is_straight_risky_coupon_bond(self) -> None:
        r, lam, rec = 0.04, 0.03, 0.4
        bond = spec_bond(conversion_ratio=0.0, coupon_rate=0.02)
        market = common_market(lam, 0.0)
        coupons = sum(1.0 * exp(-(r + lam) * 0.5 * i) for i in range(1, 11))
        exact = coupons + risky_zero_coupon(100.0, 5.0, r, lam, rec)
        assert price(bond, market, with_floor=False).dirty_price == pytest.approx(
            exact, abs=2e-3
        )


class TestMonotonicity:
    """T5 across a spot ladder from 10% to 300% of the conversion price."""

    LADDER = (10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 160.0, 200.0, 250.0, 300.0)

    @pytest.fixture(scope="class")
    @classmethod
    def reference(cls):
        bond, terms, market = reference_convertible()
        grid = build_grid(bond, market, terms=terms)
        return bond, terms, market, grid

    def test_price_dominates_parity_and_floor(self, reference) -> None:
        bond, terms, market, grid = reference
        result = price(bond, market, terms, grid=grid)
        for s in self.LADDER:
            floor = result.floor_solution.value_at(s)
            assert result.solution.value_at(s) >= max(s, floor) - 1e-6, s

    def _hazard_bump(self, reference) -> tuple[np.ndarray, np.ndarray]:
        bond, terms, market, grid = reference
        base = solve_grid(bond, market, grid, terms)
        riskier = solve_grid(
            bond, replace(market, hazard=market.hazard.shifted(0.01)), grid, terms
        )
        return grid.spots, riskier - base

    def test_price_decreasing_in_hazard_up_to_balanced(self, reference) -> None:
        spots, change = self._hazard_bump(reference)
        idx = [int(np.argmin(abs(spots - s))) for s in self.LADDER if s <= 100.0]
        assert (change[idx] <= 0).all()

    def test_price_increasing_in_hazard_when_equity_like(self, reference) -> None:
        # A deliberate exception to the spec's T5 "∂V/∂λ0 ≤ 0": surviving
        # stock drifts at r + ηλ, so the conversion option gains from λ
        # (T3: a call under JTD is BS with r → r + λ). Deep ITM that beats
        # the bond leg's loss.
        spots, change = self._hazard_bump(reference)
        idx = [int(np.argmin(abs(spots - s))) for s in (200.0, 300.0)]
        assert (change[idx] > 0).all()

    def test_vega_nonnegative_with_stock_independent_hazard(self, reference) -> None:
        bond, terms, market, grid = reference
        market = replace(market, hazard_elasticity=0.0)
        up = solve_grid(bond, replace(market, volatility=0.26), grid, terms)
        dn = solve_grid(bond, replace(market, volatility=0.24), grid, terms)
        idx = [int(np.argmin(abs(grid.spots - s))) for s in self.LADDER]
        assert (up[idx] - dn[idx] >= -1e-6).all()

    def test_vega_turns_negative_deep_busted_when_hazard_is_convex(
        self, reference
    ) -> None:
        # Not in the spec's T5, and a deliberate exception to it: with
        # p > 0, λ ∝ S^-p is convex, so more vol raises E[λ] (Jensen).
        # Deep in the busted zone the bond is almost all credit, and
        # that effect beats the option's vega.
        bond, terms, market, grid = reference
        up = solve_grid(bond, replace(market, volatility=0.26), grid, terms)
        dn = solve_grid(bond, replace(market, volatility=0.24), grid, terms)
        i = int(np.argmin(abs(grid.spots - 10.0)))
        assert up[i] < dn[i]


# Spec §08: spot → (price pts, delta % parity, bond floor pts)
REFERENCE_TABLE = {
    40.0: (79.38, 0.464, 76.81),
    60.0: (89.44, 0.556, 80.48),
    100.0: (116.45, 0.780, 84.03),
    160.0: (168.90, 0.937, 86.34),
}


class TestReferenceRun:
    """Spec §08 table, prototype grid, within 0.05 pts and 0.005 delta."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        bond, terms, market = reference_convertible()
        return price(bond, market, terms, PROTOTYPE_GRID)

    def test_headline_numbers(self, result) -> None:
        assert result.clean_points == pytest.approx(116.44, abs=0.05)
        assert result.parity == pytest.approx(100.0)
        assert result.conversion_premium == pytest.approx(0.164, abs=0.001)
        assert result.bond_floor == pytest.approx(84.03, abs=0.05)
        assert result.delta_pct_parity == pytest.approx(0.780, abs=0.005)
        assert result.gamma == pytest.approx(0.0044, abs=0.0001)

    @pytest.mark.parametrize("spot", list(REFERENCE_TABLE))
    def test_spot_ladder(self, result, spot: float) -> None:
        price_pts, delta, floor = REFERENCE_TABLE[spot]
        assert result.solution.value_at(spot) == pytest.approx(price_pts, abs=0.05)
        assert result.solution.delta_at(spot) == pytest.approx(delta, abs=0.005)
        assert result.floor_solution.value_at(spot) == pytest.approx(floor, abs=0.05)

    def test_default_grid_agrees_with_prototype_grid(self, result) -> None:
        bond, terms, market = reference_convertible()
        fast = price(bond, market, terms, with_floor=False)
        assert fast.dirty_price == pytest.approx(result.dirty_price, abs=0.02)
        assert fast.delta == pytest.approx(result.delta, abs=0.001)


class TestConvergence:
    """Halving ΔS and Δt moves price < 0.02 pts and delta < 0.001 (default grid)."""

    @pytest.mark.parametrize("spot", [40.0, 100.0, 160.0])
    def test_grid_halving(self, spot: float) -> None:
        bond, terms, market = reference_convertible()
        market = replace(market, spot=spot, hazard=HazardCurve.flat(0.03, spot))
        coarse = price(bond, market, terms, GridSpec(), with_floor=False)
        fine = price(bond, market, terms, GridSpec().refined(), with_floor=False)
        assert abs(fine.dirty_points - coarse.dirty_points) < 0.02
        assert abs(fine.delta - coarse.delta) < 0.001


def _no_credit_jtd(market) -> JTDMarketData:
    return JTDMarketData(
        spot=market.spot,
        volatility=market.volatility,
        risk_free_rate=market.risk_free_rate,
        hazard=HazardCurve.flat(0.0, market.spot),
        hazard_elasticity=0.0,
        dividend_yield=market.dividend_yield,
    )


class TestAgainstTF:
    """Where the TF tree and the JTD PDE must agree.

    With no credit risk both models collapse to the same convertible;
    with κ = 0 and zero recovery the JTD bond floor is TF's straight bond
    discounted at ``r + λ``. TF call prices carry no accrued, hence
    ``prices_are_clean=False``. TF uses 2000 steps; residual is tree error.
    """

    TERMS = JTDTerms(prices_are_clean=False)

    @pytest.mark.parametrize("example", [equity_like, hybrid, busted])
    @pytest.mark.parametrize("callable_", [False, True])
    def test_no_credit_limit(self, example, callable_: bool) -> None:
        bond, market = example()
        if callable_:
            bond = replace(bond, call_schedule={2.0: 1050.0})
        tf = tf_price(bond, replace(market, credit_spread=0.0), 2000).price
        jtd = price(
            bond, _no_credit_jtd(market), self.TERMS, with_floor=False
        ).dirty_price
        assert jtd == pytest.approx(tf, abs=0.15)

    @pytest.mark.parametrize("example", [equity_like, hybrid, busted])
    def test_straight_bond_with_zero_recovery(self, example) -> None:
        bond, market = example()
        bond = replace(bond, conversion_ratio=0.0)
        jtd_market = replace(
            _no_credit_jtd(market),
            hazard=HazardCurve.flat(market.credit_spread, market.spot),
            recovery=0.0,
        )
        tf = tf_price(bond, market, 2000).price
        assert price(bond, jtd_market, with_floor=False).dirty_price == pytest.approx(
            tf, abs=0.01
        )

    def test_coupon_date_exercise_ordering_matters_only_with_calls_on_coupon_dates(
        self,
    ) -> None:
        # TF adds the coupon, then checks exercise; JTD's default matches.
        # Flipping the flag reintroduces a gap of order one coupon.
        bond, market = hybrid()
        bond = replace(bond, call_schedule={2.0: 1050.0})
        tf = tf_price(bond, replace(market, credit_spread=0.0), 2000).price
        ex_coupon = replace(self.TERMS, exercise_before_coupon=False)
        jtd = price(
            bond, _no_credit_jtd(market), ex_coupon, with_floor=False
        ).dirty_price
        assert jtd - tf > 5.0


class TestEvents:
    @pytest.fixture
    def base(self):
        bond, terms, market = reference_convertible()
        return bond, terms, market

    def _price(self, bond, terms, market) -> float:
        return price(bond, market, terms, with_floor=False).dirty_price

    def test_bermudan_put_adds_value(self, base) -> None:
        bond, terms, market = base
        with_put = replace(bond, put_schedule={2.0: 100.0})
        assert self._price(with_put, terms, market) > self._price(bond, terms, market)

    def test_put_only_binds_on_its_date(self, base) -> None:
        # A put at 2.0 is worth less than a put at 2.0 and 3.0.
        bond, terms, market = base
        one = replace(bond, put_schedule={2.0: 100.0})
        two = replace(bond, put_schedule={2.0: 100.0, 3.0: 100.0})
        assert self._price(two, terms, market) > self._price(one, terms, market)

    def test_put_at_maturity_is_rejected(self, base) -> None:
        bond, terms, market = base
        with pytest.raises(ValueError):
            self._price(replace(bond, put_schedule={5.0: 100.0}), terms, market)

    def test_higher_trigger_is_worth_more_to_holder(self, base) -> None:
        bond, terms, market = base
        lo = self._price(bond, replace(terms, call_trigger=1.1), market)
        hi = self._price(bond, replace(terms, call_trigger=1.5), market)
        assert hi > lo

    def test_notice_period_adds_value_on_hard_call(self, base) -> None:
        bond, _, market = base
        hard = JTDTerms(call_trigger=0.0)
        no_notice = self._price(bond, hard, market)
        notice = self._price(bond, replace(hard, call_notice=30 / 365), market)
        assert notice > no_notice

    def test_discrete_dividend_lowers_value(self, base) -> None:
        bond, terms, market = base
        with_div = replace(market, discrete_dividends=((1.0, 3.0), (2.0, 3.0)))
        assert self._price(bond, terms, with_div) < self._price(bond, terms, market)

    def test_zero_dividend_is_noop(self, base) -> None:
        bond, terms, market = base
        with_div = replace(market, discrete_dividends=((1.0, 0.0),))
        assert self._price(bond, terms, with_div) == pytest.approx(
            self._price(bond, terms, market), abs=1e-3
        )

    def test_clean_call_price_worth_more_than_dirty_flat(self) -> None:
        # Hard call at 101: "clean" pays accrued on top, so it's the better deal for the holder.
        bond, _, market = reference_convertible()
        bond = replace(bond, call_schedule={1.25: 101.0})
        clean = self._price(bond, JTDTerms(prices_are_clean=True), market)
        flat = self._price(bond, JTDTerms(prices_are_clean=False), market)
        assert clean > flat

    def test_exercise_before_coupon_costs_holder(self, base) -> None:
        bond, terms, market = base
        before = self._price(bond, terms, market)
        after = self._price(bond, replace(terms, exercise_before_coupon=False), market)
        assert after > before


class TestRecovery:
    def test_par_plus_accrued_worth_more_than_par(self) -> None:
        bond, terms, market = reference_convertible()
        pa = replace(market, recovery_convention=RecoveryConvention.PAR_PLUS_ACCRUED)
        assert (
            price(bond, pa, terms, with_floor=False).dirty_price
            > price(bond, market, terms, with_floor=False).dirty_price
        )

    def test_market_value_recovery_straight_bond(self) -> None:
        # RMV folds into the discount rate: zero bond at r + λ(1 − R).
        market = replace(
            common_market(0.03, 0.0),
            recovery_convention=RecoveryConvention.MARKET_VALUE,
        )
        result = price(spec_bond(conversion_ratio=0.0), market, with_floor=False)
        assert result.dirty_price == pytest.approx(
            100 * exp(-(0.04 + 0.03 * 0.6) * 5), abs=1e-3
        )
        assert result.default_value == pytest.approx(0.4 * result.dirty_price)


class TestAccrued:
    def test_accrual_is_linear_and_resets_on_coupon_dates(self) -> None:
        accrued = accrued_interest(spec_bond(coupon_rate=0.02))
        assert accrued(0.0) == pytest.approx(0.0)
        assert accrued(0.25) == pytest.approx(0.5)
        assert accrued(0.5) == pytest.approx(0.0)

    def test_clean_is_dirty_minus_accrued_mid_period(self) -> None:
        bond = spec_bond(coupon_rate=0.02, maturity=4.75)
        result = price(bond, common_market(0.03, 1.0), with_floor=False)
        assert result.accrued == pytest.approx(0.5)
        assert result.clean_price == pytest.approx(result.dirty_price - 0.5)
