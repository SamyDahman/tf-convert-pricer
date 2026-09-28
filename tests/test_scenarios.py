"""Tests for the demo's two-model scenarios."""

import pytest

from tf_convert_pricer.jtd.calibration import price_european_option
from webapp.scenarios import (
    CALIBRATION_GRID,
    SCENARIOS,
    SOFT_CALL,
    TF_REGIMES,
    VOL_FLOOR,
    VOL_TENOR,
    atm_option_target,
    build_scenario,
)


@pytest.mark.parametrize("name", list(TF_REGIMES))
def test_jtd_hazard_calibrated_to_tf_spread(name: str) -> None:
    # Both models must see the same market credit spread.
    scenario = build_scenario(name, hazard_elasticity=1.0, recovery=0.4)
    assert scenario.cds_spread == pytest.approx(scenario.tf_market.credit_spread, abs=1e-8)
    assert scenario.jtd_market.hazard.reference_spot == scenario.tf_market.spot


@pytest.mark.parametrize("name", list(TF_REGIMES))
def test_jtd_vol_reprices_atm_option_below_bs_vol(name: str) -> None:
    # Spec §05 mode (b): σ fitted to an option priced in the JTD model,
    # not the BS vol plugged in (which would double-count default).
    scenario = build_scenario(name, hazard_elasticity=1.0, recovery=0.4)
    market = scenario.jtd_market
    model = price_european_option(market, market.spot, VOL_TENOR, True, CALIBRATION_GRID)
    target = atm_option_target(scenario.tf_market)
    assert not scenario.vol_floored
    assert model == pytest.approx(target, rel=2e-3)  # residual is grid-to-grid, not the fit
    assert market.volatility < scenario.quoted_volatility


def test_vol_floored_when_hazard_alone_overprices_the_option() -> None:
    # Busted, p = 2: an 8% spread with a steep hazard makes the ATM call
    # worth more than its 50%-vol BS price even with zero diffusion.
    scenario = build_scenario("busted", hazard_elasticity=2.0, recovery=0.4)
    assert scenario.vol_floored
    assert scenario.jtd_market.volatility == VOL_FLOOR
    assert scenario.cds_spread == pytest.approx(0.08, abs=1e-8)


def test_soft_call_is_jtd_only_and_keeps_spec_terms() -> None:
    scenario = build_scenario(SOFT_CALL, hazard_elasticity=0.5, recovery=0.3)
    assert scenario.tf_market is None
    assert scenario.quoted_volatility is None
    assert scenario.jtd_market.volatility == 0.25
    assert scenario.jtd_terms.call_trigger == 1.3
    assert scenario.jtd_market.hazard_elasticity == 0.5
    assert scenario.jtd_market.recovery == 0.3


def test_every_listed_scenario_builds() -> None:
    assert CALIBRATION_GRID.space_nodes > 0
    for name in SCENARIOS:
        assert build_scenario(name, 1.0, 0.4).bond.maturity > 0
