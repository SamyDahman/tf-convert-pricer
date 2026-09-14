"""Tests for the demo simulation module."""

import numpy as np
import pytest

from tf_convert_pricer.examples import hybrid
from webapp.simulation import compute_series, simulate_path


class TestSimulatePath:
    def test_starts_at_spot(self) -> None:
        _, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=100, seed=42)
        assert path[0] == pytest.approx(market.spot)

    def test_length_is_n_frames_plus_one(self) -> None:
        _, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=100, seed=42)
        assert len(path) == 101

    def test_all_positive(self) -> None:
        # GBM cannot produce non-positive prices.
        _, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=100, seed=42)
        assert (path > 0).all()

    def test_seed_is_reproducible(self) -> None:
        _, market = hybrid()
        p1 = simulate_path(market, n_days=252, n_frames=100, seed=42)
        p2 = simulate_path(market, n_days=252, n_frames=100, seed=42)
        assert np.array_equal(p1, p2)

    def test_different_seeds_produce_different_paths(self) -> None:
        _, market = hybrid()
        p1 = simulate_path(market, n_days=252, n_frames=100, seed=1)
        p2 = simulate_path(market, n_days=252, n_frames=100, seed=2)
        assert not np.array_equal(p1, p2)


class TestComputeSeries:
    def test_produces_one_row_per_frame(self) -> None:
        bond, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=20, seed=42)
        df = compute_series(bond, market, path, tree_steps=100, n_days=252)
        assert len(df) == 21

    def test_required_columns_present(self) -> None:
        bond, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=5, seed=42)
        df = compute_series(bond, market, path, tree_steps=100, n_days=252)
        required = {
            "frame",
            "day",
            "spot",
            "remaining_maturity",
            "fair_value",
            "parity",
            "equity_component",
            "cash_only_component",
            "delta",
            "gamma",
            "vega",
            "credit_spread_sensitivity",
            "theta",
        }
        assert required.issubset(set(df.columns))

    def test_first_frame_matches_original_state(self) -> None:
        bond, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=10, seed=42)
        df = compute_series(bond, market, path, tree_steps=100, n_days=252)
        row = df.iloc[0]
        assert row["spot"] == pytest.approx(market.spot)
        assert row["remaining_maturity"] == pytest.approx(bond.maturity)
        assert row["day"] == pytest.approx(0.0)

    def test_remaining_maturity_is_constant(self) -> None:
        # We deliberately hold the bond's maturity fixed during the
        # animation — see the compute_series docstring for the reason
        # (coupon-boundary discontinuities under aging).
        bond, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=10, seed=42)
        df = compute_series(bond, market, path, tree_steps=100, n_days=252)
        assert (df["remaining_maturity"] == bond.maturity).all()

    def test_fair_value_matches_pricer_at_first_frame(self) -> None:
        # Sanity: first-frame FV should equal what price() gives on the
        # untouched bond/market pair.
        from tf_convert_pricer.pricer import price
        bond, market = hybrid()
        path = simulate_path(market, n_days=252, n_frames=5, seed=42)
        df = compute_series(bond, market, path, tree_steps=100, n_days=252)
        expected = price(bond, market, steps=100).price
        assert df.iloc[0]["fair_value"] == pytest.approx(expected, rel=1e-9)
