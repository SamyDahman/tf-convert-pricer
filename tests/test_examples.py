"""Tests for the example instruments.

Each example must land in its named regime — checked via the pricer's
equity/cash-only split at t=0 and the parity/price relationship.
"""

from tf_convert_pricer.examples import busted, equity_like, hybrid
from tf_convert_pricer.pricer import price


STEPS = 500


class TestExamples:
    def test_equity_like_is_equity_dominated(self) -> None:
        bond, market = equity_like()
        result = price(bond, market, steps=STEPS)
        assert result.equity_component / result.price > 0.9
        # Price is anchored to parity (equity-like → trades near parity).
        assert result.parity / result.price > 0.9

    def test_hybrid_has_both_components_material(self) -> None:
        bond, market = hybrid()
        result = price(bond, market, steps=STEPS)
        e_frac = result.equity_component / result.price
        b_frac = result.cash_only_component / result.price
        assert 0.2 < e_frac < 0.8
        assert 0.2 < b_frac < 0.8

    def test_busted_is_cash_dominated(self) -> None:
        bond, market = busted()
        result = price(bond, market, steps=STEPS)
        assert result.cash_only_component / result.price > 0.7
        # Bond floor supports the price well above parity.
        assert result.parity / result.price < 0.7

    def test_prices_are_positive_and_reasonable(self) -> None:
        # Sanity: all three price out to positive numbers within a
        # plausible band around face value.
        for get_example in (equity_like, hybrid, busted):
            bond, market = get_example()
            result = price(bond, market, steps=STEPS)
            assert result.price > 0
            assert 0.3 * bond.face_value < result.price < 5.0 * bond.face_value
