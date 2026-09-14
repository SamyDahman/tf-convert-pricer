"""Tests for greeks: signs and regime-dependent orderings.

Deliberately avoids exact-value assertions — tree noise plus finite-
difference truncation makes those brittle without adaptive bump
sizing. What we can assert cleanly is signs, orderings across the
three example regimes, and one anchor: equity-like delta ≈ conversion
ratio.
"""

import math

import pytest

from tf_convert_pricer.examples import busted, equity_like, hybrid
from tf_convert_pricer.greeks import (
    credit_spread_sensitivity,
    delta,
    gamma,
    theta,
    vega,
)


STEPS = 500


def _all_regimes():
    return [("equity_like", equity_like()), ("hybrid", hybrid()), ("busted", busted())]


class TestGreekSigns:
    def test_delta_positive_in_every_regime(self) -> None:
        for name, (bond, market) in _all_regimes():
            assert delta(bond, market, STEPS) > 0, name

    def test_vega_nonnegative(self) -> None:
        # Vega can be near-zero when there's no meaningful optionality
        # left (deep-ITM converts converting at t=0), so allow zero.
        for name, (bond, market) in _all_regimes():
            assert vega(bond, market, STEPS) >= -1e-6, name

    def test_credit_spread_sensitivity_negative(self) -> None:
        # Spread up → price down for any credit-risky bond. Allow near-
        # zero for equity-like where the cash-only leg is tiny.
        for name, (bond, market) in _all_regimes():
            assert credit_spread_sensitivity(bond, market, STEPS) <= 1e-6, name

    def test_theta_is_finite(self) -> None:
        # Unlike options, bonds pull to par as time passes, so theta
        # for a convert isn't necessarily negative. The sign depends
        # on regime — see greeks.theta docstring.
        for name, (bond, market) in _all_regimes():
            assert math.isfinite(theta(bond, market, STEPS)), name

    def test_theta_positive_for_busted(self) -> None:
        # A busted convert trades at a deep discount to par (~852 vs.
        # 1000); the bond-leg pull-to-par is the dominant time effect,
        # so theta should be clearly positive.
        bond, market = busted()
        assert theta(bond, market, STEPS) > 0


class TestGreeksAcrossRegimes:
    def test_delta_orders_by_moneyness(self) -> None:
        # Equity-like > hybrid > busted — bond's stock-sensitivity
        # grows as parity climbs relative to the bond floor.
        d_eq = delta(*equity_like(), steps=STEPS)
        d_hy = delta(*hybrid(), steps=STEPS)
        d_bu = delta(*busted(), steps=STEPS)
        assert d_eq > d_hy > d_bu

    def test_equity_like_delta_near_conversion_ratio(self) -> None:
        # Deep ITM with early conversion optimal — dV/dS ≈ conversion
        # ratio (each dollar of spot buys another `c` dollars of parity).
        bond, market = equity_like()
        assert 9.0 < delta(bond, market, STEPS) < 10.5

    def test_gamma_largest_in_hybrid(self) -> None:
        # Gamma peaks where optionality is at-the-money.
        g_eq = gamma(*equity_like(), steps=STEPS)
        g_hy = gamma(*hybrid(), steps=STEPS)
        g_bu = gamma(*busted(), steps=STEPS)
        assert g_hy > g_eq
        assert g_hy > g_bu

    def test_credit_spread_sensitivity_largest_in_busted(self) -> None:
        # |dV/ds| grows as the cash-only leg's share of value grows —
        # busted bond floors are what suffers when spreads widen.
        c_eq = credit_spread_sensitivity(*equity_like(), steps=STEPS)
        c_hy = credit_spread_sensitivity(*hybrid(), steps=STEPS)
        c_bu = credit_spread_sensitivity(*busted(), steps=STEPS)
        assert abs(c_bu) > abs(c_hy) > abs(c_eq)

    def test_vega_largest_in_hybrid(self) -> None:
        # Vol matters most where both branches are reachable — the
        # equity-like regime has no meaningful optionality (converted),
        # and the busted regime is dominated by credit rather than vol.
        v_eq = vega(*equity_like(), steps=STEPS)
        v_hy = vega(*hybrid(), steps=STEPS)
        v_bu = vega(*busted(), steps=STEPS)
        assert v_hy > v_eq
        assert v_hy > v_bu
