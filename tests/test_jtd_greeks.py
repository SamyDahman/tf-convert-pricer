"""Tests for JTD Greeks and the risk report (spec §07, §08)."""

from dataclasses import replace

import pytest

from tf_convert_pricer.jtd.calibration import calibrate_market
from tf_convert_pricer.jtd.examples import PROTOTYPE_GRID, reference_convertible
from tf_convert_pricer.jtd.greeks import risk_report, rolled_bond, spot_ladder
from tf_convert_pricer.jtd.grid import GridSpec
from tf_convert_pricer.jtd.market import CDSCurve, HazardCurve
from tf_convert_pricer.jtd.pricer import price

FAST = GridSpec(space_nodes=200, steps_per_year=100)


@pytest.fixture(scope="module")
def report():
    bond, terms, market = reference_convertible()
    return risk_report(bond, market, terms)


class TestReferenceRiskReport:
    """The spec's "at spot 100" table (CS01 bumps λ0 by 1bp/(1−R))."""

    def test_delta_gamma(self, report) -> None:
        assert report.delta_pct_parity == pytest.approx(0.780, abs=0.005)
        assert report.gamma == pytest.approx(0.0044, abs=0.0002)
        assert report.gamma_per_1pct == pytest.approx(report.gamma * 100.0 * 0.01)

    def test_vega_cs01_rho(self, report) -> None:
        assert report.vega == pytest.approx(0.48, abs=0.01)
        assert report.cs01 == pytest.approx(-0.0063, abs=0.0002)
        assert report.rho == pytest.approx(-0.015, abs=0.0005)

    def test_jump_to_default(self, report) -> None:
        assert report.jtd_naked == pytest.approx(-76.4, abs=0.1)
        assert report.jtd_hedged == pytest.approx(1.6, abs=0.1)

    def test_borrow_acts_exactly_like_dividend_yield(self, report) -> None:
        assert report.borrow_sensitivity == pytest.approx(report.dividend_sensitivity)
        assert report.borrow_sensitivity < 0

    def test_carry_without_dividends(self, report) -> None:
        assert report.carry.coupon_accrual == pytest.approx(2.0 / 365, rel=1e-6)
        assert report.carry.dividends_owed == 0.0
        assert report.carry.short_rebate == pytest.approx(
            report.delta * 100 * 0.04 / 365
        )

    def test_cds_hedge_adds_protection_payout(self) -> None:
        bond, terms, market = reference_convertible()
        hedged = risk_report(bond, market, terms, FAST, cds_hedge_notional=10.0)
        naked = risk_report(bond, market, terms, FAST)
        assert hedged.jtd_hedged - naked.jtd_hedged == pytest.approx(10.0 * 0.6)


class TestDeltaSplit:
    """Spec §08 ladder: Δ_eq freezes λ at each spot; the gap is credit delta."""

    @pytest.fixture(scope="class")
    @classmethod
    def ladder(cls):
        bond, terms, market = reference_convertible()
        return {
            row.spot: row
            for row in spot_ladder(
                bond, market, [40.0, 60.0, 100.0, 160.0], terms, PROTOTYPE_GRID
            )
        }

    @pytest.mark.parametrize(
        "spot,delta_eq", [(40.0, 0.318), (60.0, 0.499), (100.0, 0.767), (160.0, 0.938)]
    )
    def test_equity_delta(self, ladder, spot: float, delta_eq: float) -> None:
        assert ladder[spot].equity_delta_pct_parity == pytest.approx(
            delta_eq, abs=0.005
        )

    def test_credit_delta_large_when_busted_and_vanishes_when_equity_like(
        self, ladder
    ) -> None:
        busted = ladder[40.0].delta_pct_parity - ladder[40.0].equity_delta_pct_parity
        equity = ladder[160.0].delta_pct_parity - ladder[160.0].equity_delta_pct_parity
        assert busted == pytest.approx(0.15, abs=0.01)
        assert abs(equity) < 0.005


class TestGridDelta:
    def test_stencil_matches_cubic_difference_on_same_grid(self) -> None:
        bond, terms, market = reference_convertible()
        sol = price(bond, market, terms, with_floor=False).solution
        fd = (sol.value_at(100.25) - sol.value_at(99.75)) / 0.5
        assert sol.delta == pytest.approx(fd, abs=1e-4)

    def test_close_to_full_reprice_at_bumped_spot(self) -> None:
        # Re-solving at S ± 1 (λ0 and its reference spot held fixed — the
        # sticky-hazard convention) builds *different* grids, so node
        # placement noise of a few 1e-3 enters. This is why the spec takes
        # delta from the grid and bumped Greeks on a shared grid.
        bond, terms, market = reference_convertible()
        base = price(bond, market, terms, with_floor=False)
        up = price(
            bond, replace(market, spot=101.0), terms, with_floor=False
        ).dirty_price
        dn = price(
            bond, replace(market, spot=99.0), terms, with_floor=False
        ).dirty_price
        assert base.delta == pytest.approx((up - dn) / 2.0, abs=0.005)


class TestCalibratedReport:
    """With a CDS curve: re-bootstrap on credit, rate and recovery bumps."""

    @pytest.fixture(scope="class")
    @classmethod
    def calibrated(cls):
        bond, terms, _ = reference_convertible()
        _, _, market = reference_convertible()
        cds = CDSCurve((1.0, 3.0, 5.0), (0.015, 0.017, 0.018))
        market = calibrate_market(market, cds, FAST)
        return risk_report(bond, market, terms, FAST)

    def test_buckets_sum_to_parallel(self, calibrated) -> None:
        total = sum(v for _, v in calibrated.cs01_buckets)
        assert len(calibrated.cs01_buckets) == 3
        assert total == pytest.approx(calibrated.cs01, rel=0.05)

    def test_signs(self, calibrated) -> None:
        assert calibrated.cs01 < 0
        assert calibrated.rho < 0
        assert calibrated.vega > 0


class TestTheta:
    def test_roll_across_coupon_date_has_no_cliff(self) -> None:
        # A coupon paid inside the roll window is added back to theta,
        # so rolling across it doesn't show a −1 point drop.
        bond, terms, market = reference_convertible()
        bond = replace(bond, maturity=5.0 - 0.5 + 0.5 / 365)
        report = risk_report(bond, market, terms, FAST)
        assert abs(report.theta) < 0.05

    def test_rolled_bond_shifts_schedules(self) -> None:
        bond, _, _ = reference_convertible()
        bond = replace(bond, put_schedule={2.0: 100.0, 0.001: 100.0})
        rolled, paid = rolled_bond(bond, 1 / 365)
        assert rolled.maturity == pytest.approx(5.0 - 1 / 365)
        assert list(rolled.call_schedule) == [pytest.approx(3.0 - 1 / 365)]
        assert list(rolled.put_schedule) == [pytest.approx(2.0 - 1 / 365)]
        assert paid == 0.0

    def test_theta_finite_and_small_for_reference(self) -> None:
        bond, terms, market = reference_convertible()
        market = replace(market, hazard=HazardCurve.flat(0.03, 100.0))
        report = risk_report(bond, market, terms, FAST)
        assert abs(report.theta) < 0.05
