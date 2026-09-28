"""Tests for JTD market inputs: hazard curve lookup and λ(S, t)."""

import numpy as np
import pytest

from tf_convert_pricer.jtd.market import CDSCurve, HazardCurve, JTDMarketData


def make_market(**overrides) -> JTDMarketData:
    defaults = dict(
        spot=100.0,
        volatility=0.25,
        risk_free_rate=0.04,
        hazard=HazardCurve.flat(0.03, reference_spot=100.0),
        hazard_elasticity=1.0,
    )
    defaults.update(overrides)
    return JTDMarketData(**defaults)


class TestHazardCurve:
    def test_buckets_are_right_closed_with_flat_extrapolation(self) -> None:
        curve = HazardCurve((1.0, 3.0), (0.01, 0.02), reference_spot=100.0)
        assert curve.base_intensity(0.0) == 0.01
        assert curve.base_intensity(1.0) == 0.01
        assert curve.base_intensity(1.0 + 1e-6) == 0.02
        assert curve.base_intensity(10.0) == 0.02

    def test_rejects_unsorted_pillars(self) -> None:
        with pytest.raises(ValueError):
            HazardCurve((3.0, 1.0), (0.01, 0.02), reference_spot=100.0)

    def test_shift_floors_at_zero(self) -> None:
        curve = HazardCurve.flat(0.001, 100.0).shifted(-0.01)
        assert curve.intensities == (0.0,)


class TestIntensity:
    def test_equals_base_at_reference_spot(self) -> None:
        assert make_market().intensity(np.array([100.0]), 0.0)[0] == pytest.approx(0.03)

    def test_power_law_in_spot(self) -> None:
        # p = 1: halving the stock doubles the hazard.
        lam = make_market().intensity(np.array([50.0, 200.0]), 0.0)
        assert lam == pytest.approx([0.06, 0.015])

    def test_capped_at_lambda_max_including_zero_spot(self) -> None:
        lam = make_market(lambda_max=2.0).intensity(np.array([0.0, 1.0]), 0.0)
        assert lam == pytest.approx([2.0, 2.0])

    def test_p_zero_is_flat(self) -> None:
        lam = make_market(hazard_elasticity=0.0).intensity(
            np.array([0.0, 10.0, 500.0]), 0.0
        )
        assert lam == pytest.approx([0.03] * 3)

    def test_zero_base_hazard_is_zero_everywhere(self) -> None:
        market = make_market(hazard=HazardCurve.flat(0.0, 100.0))
        assert not market.intensity(np.array([0.0, 50.0]), 0.0).any()

    def test_reference_spot_does_not_follow_market_spot(self) -> None:
        # Sticky hazard: moving spot moves along λ(S), λ0 stays anchored.
        market = make_market(spot=50.0)
        assert market.intensity(np.array([50.0]), 0.0)[0] == pytest.approx(0.06)


class TestValidation:
    @pytest.mark.parametrize(
        "field,value",
        [("jump_size", 0.0), ("hazard_elasticity", -1.0), ("recovery", 1.0)],
    )
    def test_rejects_out_of_range(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            make_market(**{field: value})

    def test_cds_single_pillar_shift(self) -> None:
        curve = CDSCurve((1.0, 5.0), (0.01, 0.02)).shifted(0.001, pillar=1)
        assert curve.spreads == pytest.approx((0.01, 0.021))
