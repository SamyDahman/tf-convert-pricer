"""Tests for JTD calibration: CDS bootstrap, implied vols, options in the model."""

from dataclasses import replace
from math import exp

import pytest

from tf_convert_pricer.jtd.analytic import black_scholes_call
from tf_convert_pricer.jtd.calibration import (
    bootstrap_hazard,
    calibrate_market,
    convert_implied_volatility,
    model_par_spread,
    option_implied_volatility,
    price_european_option,
)
from tf_convert_pricer.jtd.examples import (
    PROTOTYPE_GRID,
    common_market,
    reference_convertible,
)
from tf_convert_pricer.jtd.grid import GridSpec
from tf_convert_pricer.jtd.market import CDSCurve, HazardCurve
from tf_convert_pricer.jtd.pricer import price

FAST = GridSpec(space_nodes=200, steps_per_year=100)
CDS = CDSCurve((1.0, 3.0, 5.0), (0.015, 0.017, 0.018))


class TestEuropeanOptions:
    def test_t3_call_is_black_scholes_with_shifted_rate(self) -> None:
        market = common_market(intensity=0.03, hazard_elasticity=0.0)
        model = price_european_option(market, 100.0, 5.0, True, PROTOTYPE_GRID)
        exact = black_scholes_call(100.0, 100.0, 5.0, 0.07, 0.25)
        assert round(model, 4) == round(exact, 4) == 36.9563
        assert black_scholes_call(100.0, 100.0, 5.0, 0.04, 0.25) == pytest.approx(
            30.3127, abs=1e-4
        )

    def test_put_call_parity_under_default(self) -> None:
        # η = 1: the stock forward is still S e^{-qT} (the +λ drift pays
        # for the jump), so ordinary parity holds, with the put receiving
        # the discounted strike on default.
        market = common_market(intensity=0.03, hazard_elasticity=1.0)
        call = price_european_option(market, 100.0, 2.0, True)
        put = price_european_option(market, 100.0, 2.0, False)
        assert call - put == pytest.approx(100.0 - 100.0 * exp(-0.08), abs=2e-3)

    def test_option_implied_vol_roundtrip(self) -> None:
        market = common_market(intensity=0.03, hazard_elasticity=1.0)
        target = price_european_option(
            replace(market, volatility=0.31), 90.0, 1.0, False, FAST
        )
        vol = option_implied_volatility(market, 90.0, 1.0, False, target, FAST)
        assert vol == pytest.approx(0.31, abs=1e-3)


class TestHazardBootstrap:
    def test_flat_hazard_without_elasticity_gives_textbook_spread(self) -> None:
        # p = 0, constant λ, continuous premium: s = λ(1 − R) exactly.
        market = common_market(intensity=0.03, hazard_elasticity=0.0)
        assert model_par_spread(market, 5.0, FAST) == pytest.approx(0.018, abs=1e-6)

    def test_bootstrap_reprices_every_pillar(self) -> None:
        market = calibrate_market(common_market(hazard_elasticity=1.0), CDS, FAST)
        for pillar, spread in zip(CDS.pillars, CDS.spreads, strict=True):
            assert model_par_spread(market, pillar, FAST) == pytest.approx(
                spread, abs=1e-9
            )
        assert market.cds_curve == CDS
        assert market.hazard.reference_spot == market.spot

    @pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
    def test_naive_shortcut_is_off_once_hazard_depends_on_stock(self, p: float) -> None:
        # Spec §05 says s/(1 − R) *overstates* with p > 0. The sign isn't
        # robust: Jensen (λ ∝ S^-p convex) pushes E[λ] up, but survivors
        # drift up at +ηλ to where λ is lower. Here p = 1 needs a λ0 above
        # the shortcut and p = 2 one below. Only "it's wrong" is general.
        flat = CDSCurve((5.0,), (0.018,))
        exact = bootstrap_hazard(common_market(hazard_elasticity=0.0), flat, FAST)
        assert exact.intensities[0] == pytest.approx(0.018 / 0.6, abs=1e-9)
        hazard = bootstrap_hazard(common_market(hazard_elasticity=p), flat, FAST)
        assert abs(hazard.intensities[0] - 0.018 / 0.6) > 1e-4

    def test_warm_start_gives_same_curve(self) -> None:
        market = common_market(hazard_elasticity=1.0)
        cold = bootstrap_hazard(market, CDS, FAST)
        warm = bootstrap_hazard(market, CDS, FAST, initial=cold.shifted(0.001))
        assert warm.intensities == pytest.approx(cold.intensities, abs=1e-9)


class TestConvertImpliedVol:
    def test_roundtrip(self) -> None:
        bond, terms, market = reference_convertible()
        target = price(
            bond, replace(market, volatility=0.30), terms, FAST, with_floor=False
        ).dirty_price
        vol = convert_implied_volatility(bond, market, target, terms, FAST)
        assert vol == pytest.approx(0.30, abs=2e-3)

    def test_hazard_curve_input_is_untouched(self) -> None:
        # Sanity: σ calibration never re-bootstraps credit.
        bond, terms, market = reference_convertible()
        market = replace(market, hazard=HazardCurve.flat(0.05, 100.0))
        target = price(bond, market, terms, FAST, with_floor=False).dirty_price
        assert convert_implied_volatility(
            bond, market, target, terms, FAST
        ) == pytest.approx(0.25, abs=2e-3)
