"""Tests for the demo's JTD path simulation and curve-based series."""

from dataclasses import replace

import numpy as np
import pytest

from tf_convert_pricer.jtd.examples import reference_convertible
from tf_convert_pricer.jtd.greeks import risk_report
from tf_convert_pricer.jtd.market import HazardCurve
from webapp.jtd_simulation import (
    SimulatedPath,
    compute_jtd_series,
    jtd_profile,
    simulate_jtd_path,
    solve_curves,
    tf_profile,
)
from webapp.scenarios import build_scenario


@pytest.fixture(scope="module")
def reference():
    bond, terms, market = reference_convertible()
    return bond, terms, market, solve_curves(bond, market, terms)


class TestSimulateJTDPath:
    def test_starts_at_spot_and_is_reproducible(self) -> None:
        _, _, market = reference_convertible()
        a = simulate_jtd_path(market, 252, 100, seed=3)
        b = simulate_jtd_path(market, 252, 100, seed=3)
        assert a.spots[0] == market.spot
        assert len(a.spots) == 101
        assert np.array_equal(a.spots, b.spots) and a.default_frame == b.default_frame

    def test_no_default_when_disallowed_or_hazard_zero(self) -> None:
        _, _, market = reference_convertible()
        assert simulate_jtd_path(market, 252, 100, seed=1, allow_default=False).default_frame is None
        safe = replace(market, hazard=HazardCurve.flat(0.0, 100.0))
        assert all(simulate_jtd_path(safe, 252, 50, seed=s).default_frame is None for s in range(20))

    def test_default_sends_stock_to_zero_and_keeps_it_there(self) -> None:
        _, _, market = reference_convertible()
        # λ = 10 flat: survival over a year is e^{-10}, so this path defaults.
        risky = replace(market, hazard=HazardCurve.flat(10.0, 100.0), hazard_elasticity=0.0, lambda_max=10.0)
        path = simulate_jtd_path(risky, 252, 100, seed=0)
        assert path.default_frame is not None
        assert (path.spots[path.default_frame :] == 0.0).all()
        assert (path.spots[: path.default_frame] > 0.0).all()

    def test_default_frequency_matches_hazard(self) -> None:
        # p = 0, λ = 20%: P(default within 1y) = 1 − e^{−0.2} ≈ 18%.
        _, _, market = reference_convertible()
        flat = replace(market, hazard=HazardCurve.flat(0.2, 100.0), hazard_elasticity=0.0)
        rate = np.mean([simulate_jtd_path(flat, 252, 50, s, substeps=1).default_frame is not None for s in range(1000)])
        assert rate == pytest.approx(1 - np.exp(-0.2), abs=0.035)


class TestCurves:
    def test_values_at_spot_match_risk_report(self, reference) -> None:
        # Same grid conventions as jtd.greeks, so t=0 frames equal the report.
        bond, terms, market, curves = reference
        report = risk_report(bond, market, terms)
        df = compute_jtd_series(bond, market, curves, SimulatedPath(np.array([100.0]), None), 252)
        row = df.iloc[0]
        assert row.fair_value == pytest.approx(report.pricing.dirty_price, abs=1e-9)
        assert row.delta == pytest.approx(report.delta_pct_parity, abs=1e-9)
        assert row.vega == pytest.approx(report.vega, abs=1e-9)
        assert row.credit_spread_sensitivity == pytest.approx(report.cs01, abs=1e-9)
        assert row.theta == pytest.approx(report.theta, abs=1e-9)
        assert row.equity_delta == pytest.approx(report.equity_delta, abs=0.01)
        assert row.jtd_naked == pytest.approx(report.jtd_naked, abs=1e-9)

    def test_post_default_frames_pay_recovery_with_zero_greeks(self, reference) -> None:
        bond, _, market, curves = reference
        path = SimulatedPath(np.array([100.0, 90.0, 0.0, 0.0]), default_frame=2)
        df = compute_jtd_series(bond, market, curves, path, 3)
        assert df.defaulted.tolist() == [False, False, True, True]
        assert (df.fair_value.iloc[2:] == 40.0).all()
        assert (df[["delta", "gamma", "vega", "theta"]].iloc[2:] == 0.0).all().all()

    def test_profile_is_above_parity_and_floor(self, reference) -> None:
        bond, _, market, curves = reference
        prof = jtd_profile(bond, market, curves, np.linspace(10.0, 300.0, 30))
        assert (prof.value >= np.maximum(prof.parity, prof.floor) - 1e-6).all()


def test_tf_profile_floor_is_flat_and_value_increasing() -> None:
    sc = build_scenario("hybrid", 1.0, 0.4)
    prof = tf_profile(sc.bond, sc.tf_market, np.array([50.0, 100.0, 150.0]), steps=200)
    assert prof.floor.nunique() == 1
    assert prof.value.is_monotonic_increasing
