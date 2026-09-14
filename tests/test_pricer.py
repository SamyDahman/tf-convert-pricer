"""Tests for the TF pricer.

Layered per the staged build in pricer.py — each stage's tests should
pass before moving to the next. Start with the sanity checks from
docs/MODEL.md before anything else once we get to the priced bond:
  - credit_spread -> 0 converges toward a standard convertible
  - conversion_ratio -> 0 converges toward a plain credit-risky bond
"""

import numpy as np
import pytest

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.market import MarketData
from tf_convert_pricer.pricer import (
    StockTree,
    coupon_schedule,
    price,
    terminal_payoff,
    terminal_payoff_split,
)


def make_bond(**overrides) -> ConvertibleBond:
    """Convenience factory: bond with sensible defaults, override any field.

    Default coupon rate is zero — this makes the factory produce the
    simplest possible bond for tests that don't care about coupons.
    Coupon-sensitive tests should set ``coupon_rate`` explicitly.
    """
    defaults = dict(
        face_value=1000.0,
        coupon_rate=0.0,
        coupon_frequency=2,
        maturity=5.0,
        conversion_ratio=10.0,
    )
    defaults.update(overrides)
    return ConvertibleBond(**defaults)


class TestStockTree:
    """Step 1: bare CRR stock price tree, no bond logic yet."""

    @pytest.fixture
    def tree(self) -> StockTree:
        return StockTree(
            spot=100.0,
            volatility=0.3,
            risk_free_rate=0.05,
            dividend_yield=0.02,
            maturity=1.0,
            steps=100,
        )

    def test_up_and_down_are_reciprocals(self, tree: StockTree) -> None:
        assert tree.u * tree.d == pytest.approx(1.0)

    def test_risk_neutral_probability_in_unit_interval(self, tree: StockTree) -> None:
        assert 0.0 < tree.p < 1.0

    def test_martingale_property(self, tree: StockTree) -> None:
        # Under the risk-neutral measure, the one-step expected gross
        # return equals exp((r - q) dt). This is the identity that the
        # p formula is derived from — the test verifies the derivation
        # holds numerically end-to-end.
        expected_growth = np.exp((tree.risk_free_rate - tree.dividend_yield) * tree.dt)
        one_step_expectation = tree.p * tree.u + (1.0 - tree.p) * tree.d
        assert one_step_expectation == pytest.approx(expected_growth)

    def test_root_level_is_spot(self, tree: StockTree) -> None:
        prices = tree.prices_at(0)
        assert prices.shape == (1,)
        assert prices[0] == pytest.approx(tree.spot)

    def test_level_shape(self, tree: StockTree) -> None:
        for step in [0, 1, 50, tree.steps]:
            assert tree.prices_at(step).shape == (step + 1,)

    def test_middle_node_at_even_step_is_spot(self, tree: StockTree) -> None:
        # One up + one down (in any order) returns to spot in a
        # recombining CRR tree, so the middle node at any even step
        # equals spot exactly.
        prices = tree.prices_at(2)
        assert prices[1] == pytest.approx(tree.spot)

    def test_extreme_up_and_down_nodes_at_terminal(self, tree: StockTree) -> None:
        prices = tree.prices_at(tree.steps)
        assert prices[-1] == pytest.approx(tree.spot * tree.u**tree.steps)
        assert prices[0] == pytest.approx(tree.spot * tree.d**tree.steps)

    def test_prices_at_out_of_range_raises(self, tree: StockTree) -> None:
        with pytest.raises(ValueError):
            tree.prices_at(-1)
        with pytest.raises(ValueError):
            tree.prices_at(tree.steps + 1)


class TestTerminalPayoff:
    """Step 2: terminal payoffs at maturity, before any induction."""

    def test_deep_itm_pays_parity(self) -> None:
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0)
        # Parity at S = 500 is 5000 ≫ face → convert everywhere.
        terminal_prices = np.array([500.0, 600.0, 700.0])
        payoff = terminal_payoff(bond, terminal_prices)
        assert payoff == pytest.approx(np.array([5000.0, 6000.0, 7000.0]))

    def test_deep_otm_pays_redemption(self) -> None:
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0)
        # Parity at S = 10 is 100 ≪ face → redeem everywhere.
        terminal_prices = np.array([10.0, 20.0, 30.0])
        payoff = terminal_payoff(bond, terminal_prices)
        assert payoff == pytest.approx(np.array([1000.0, 1000.0, 1000.0]))

    def test_mixed_regime_selects_per_node(self) -> None:
        # Conversion ratio 10 on a $1000-face bond puts the crossover at
        # S = 100. A range spanning it exercises the per-node max.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0)
        terminal_prices = np.array([50.0, 100.0, 150.0])
        payoff = terminal_payoff(bond, terminal_prices)
        assert payoff == pytest.approx(np.array([1000.0, 1000.0, 1500.0]))

    def test_custom_redemption_overrides_face(self) -> None:
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, redemption_value=1100.0)
        terminal_prices = np.array([10.0])  # deep OTM
        payoff = terminal_payoff(bond, terminal_prices)
        assert payoff == pytest.approx(np.array([1100.0]))

    def test_shape_matches_input(self) -> None:
        bond = make_bond()
        terminal_prices = np.array([50.0, 100.0, 150.0, 200.0])
        payoff = terminal_payoff(bond, terminal_prices)
        assert payoff.shape == terminal_prices.shape


class TestTerminalPayoffSplit:
    """Step 4a: TF equity/cash-only split at terminal nodes."""

    def test_deep_itm_all_equity(self) -> None:
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0)
        terminal_prices = np.array([500.0, 600.0, 700.0])
        equity, cash_only = terminal_payoff_split(bond, terminal_prices)
        assert equity == pytest.approx(np.array([5000.0, 6000.0, 7000.0]))
        assert cash_only == pytest.approx(np.zeros(3))

    def test_deep_otm_all_cash(self) -> None:
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0)
        terminal_prices = np.array([10.0, 20.0, 30.0])
        equity, cash_only = terminal_payoff_split(bond, terminal_prices)
        assert equity == pytest.approx(np.zeros(3))
        assert cash_only == pytest.approx(np.array([1000.0, 1000.0, 1000.0]))

    def test_mixed_regime_per_node(self) -> None:
        # Crossover at S = 100 (c=10, face=1000).
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0)
        terminal_prices = np.array([50.0, 100.0, 150.0])
        equity, cash_only = terminal_payoff_split(bond, terminal_prices)
        # S=50: redeem (100 < 1000); S=100: tie, convert per convention;
        # S=150: convert (1500 > 1000).
        assert equity == pytest.approx(np.array([0.0, 1000.0, 1500.0]))
        assert cash_only == pytest.approx(np.array([1000.0, 0.0, 0.0]))

    def test_split_sums_to_total_payoff(self) -> None:
        # Structural invariant: E + B == max(redemption, parity).
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0)
        terminal_prices = np.array([50.0, 100.0, 150.0, 500.0])
        equity, cash_only = terminal_payoff_split(bond, terminal_prices)
        assert equity + cash_only == pytest.approx(terminal_payoff(bond, terminal_prices))

    def test_custom_redemption_moves_the_boundary(self) -> None:
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, redemption_value=2000.0)
        # Boundary shifts: convert only when parity ≥ 2000, i.e. S ≥ 200.
        terminal_prices = np.array([150.0, 200.0, 250.0])
        equity, cash_only = terminal_payoff_split(bond, terminal_prices)
        assert equity == pytest.approx(np.array([0.0, 2000.0, 2500.0]))
        assert cash_only == pytest.approx(np.array([2000.0, 0.0, 0.0]))


class TestPriceSingleRate:
    """Step 3: single-rate backward induction (no credit split yet).

    All expected values assume the pricer discounts everything at the
    risk-free rate — which is the ``credit_spread → 0`` limit of the
    step-4 TF split.
    """

    def test_zero_conversion_ratio_is_zero_coupon_bond(self) -> None:
        # No conversion option → the "bond" is a plain zero-coupon
        # note: value = redemption × exp(-r × T).
        bond = make_bond(face_value=1000.0, conversion_ratio=0.0, maturity=5.0)
        market = MarketData(
            spot=100.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.0,
        )
        expected = 1000.0 * np.exp(-0.05 * 5.0)
        assert price(bond, market, steps=500).price == pytest.approx(expected, rel=1e-4)

    def test_deep_itm_zero_dividend_recovers_parity(self) -> None:
        # Deep ITM at every terminal node → payoff = parity throughout.
        # E_Q[c × S_T] × exp(-r T) = c × S_0 × exp(-q T); with q = 0 that
        # equals parity today.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=1.0)
        market = MarketData(
            spot=10_000.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.0,
        )
        expected = 10.0 * 10_000.0
        assert price(bond, market, steps=500).price == pytest.approx(expected, rel=1e-3)

    def test_deep_itm_with_dividend_yield_converts_early(self) -> None:
        # With q > 0 and no coupons to compensate, the "hold" branch
        # bleeds dividends the holder doesn't receive. Step 5b's
        # voluntary conversion detects this and converts at t=0, so
        # price = parity today, not c × S_0 × exp(-q T). Same test
        # produced 188k before step 5b; after, it produces 200k.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=2.0)
        market = MarketData(
            spot=10_000.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.03,
        )
        expected = 10.0 * 10_000.0
        assert price(bond, market, steps=500).price == pytest.approx(expected, rel=1e-3)

    def test_deep_otm_recovers_bond_floor(self) -> None:
        # Deep OTM at every terminal node → payoff = redemption
        # throughout; discounting gives redemption × exp(-r × T).
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=5.0)
        market = MarketData(
            spot=0.01,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.0,
        )
        expected = 1000.0 * np.exp(-0.05 * 5.0)
        assert price(bond, market, steps=500).price == pytest.approx(expected, rel=1e-4)

    def test_parity_field_is_ratio_times_spot(self) -> None:
        bond = make_bond(conversion_ratio=7.5)
        market = MarketData(
            spot=137.0,
            volatility=0.2,
            risk_free_rate=0.04,
            credit_spread=0.0,
            dividend_yield=0.0,
        )
        assert price(bond, market, steps=100).parity == pytest.approx(7.5 * 137.0)

    def test_price_is_stable_across_step_counts(self) -> None:
        # A hybrid case (not at a boundary regime) — prices at N=500
        # and N=1000 should agree to a few tens of bps.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=2.0)
        market = MarketData(
            spot=100.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.02,
        )
        p500 = price(bond, market, steps=500).price
        p1000 = price(bond, market, steps=1000).price
        assert abs(p1000 - p500) / p500 < 1e-3


class TestPriceWithCreditSpread:
    """Step 4b: TF split with a positive credit spread."""

    def test_deep_otm_recovers_credit_risky_bond_floor(self) -> None:
        # OTM everywhere → 100% cash-only component; discount at r + s.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=5.0)
        market = MarketData(
            spot=0.01,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.03,
            dividend_yield=0.0,
        )
        expected = 1000.0 * np.exp(-(0.05 + 0.03) * 5.0)
        result = price(bond, market, steps=500)
        assert result.price == pytest.approx(expected, rel=1e-4)
        assert result.equity_component == pytest.approx(0.0, abs=1e-6)
        assert result.cash_only_component == pytest.approx(expected, rel=1e-4)

    def test_deep_itm_price_insensitive_to_credit_spread(self) -> None:
        # ITM everywhere → 100% equity component; changing s should
        # not move the price at all.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=1.0)
        market_low = MarketData(
            spot=10_000.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.0,
        )
        market_high = MarketData(
            spot=10_000.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.10,
            dividend_yield=0.0,
        )
        p_low = price(bond, market_low, steps=500).price
        p_high = price(bond, market_high, steps=500).price
        assert p_high == pytest.approx(p_low, rel=1e-6)

    def test_higher_spread_reduces_price_in_hybrid_regime(self) -> None:
        # A middling spot where both components are meaningful — the
        # cash-only leg loses value as s grows, so total falls.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=5.0)
        market_low = MarketData(
            spot=90.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.01,
            dividend_yield=0.02,
        )
        market_high = MarketData(
            spot=90.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.05,
            dividend_yield=0.02,
        )
        assert price(bond, market_high, steps=500).price < price(bond, market_low, steps=500).price

    def test_components_sum_to_reported_price(self) -> None:
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=3.0)
        market = MarketData(
            spot=110.0,
            volatility=0.35,
            risk_free_rate=0.04,
            credit_spread=0.02,
            dividend_yield=0.01,
        )
        result = price(bond, market, steps=500)
        assert result.equity_component + result.cash_only_component == pytest.approx(result.price)

    def test_hybrid_has_both_components_positive(self) -> None:
        # In a genuinely hybrid regime, both branches are reachable
        # from the root — E and B should each be materially positive.
        bond = make_bond(face_value=1000.0, conversion_ratio=10.0, maturity=5.0)
        market = MarketData(
            spot=100.0,
            volatility=0.35,
            risk_free_rate=0.05,
            credit_spread=0.02,
            dividend_yield=0.02,
        )
        result = price(bond, market, steps=500)
        assert result.equity_component > 0.05 * result.price
        assert result.cash_only_component > 0.05 * result.price


class TestCouponSchedule:
    """Step 5a: bond → (time, amount) coupon schedule."""

    def test_semiannual_five_year_has_ten_coupons(self) -> None:
        bond = make_bond(face_value=1000.0, coupon_rate=0.05, coupon_frequency=2, maturity=5.0)
        schedule = coupon_schedule(bond)
        assert len(schedule) == 10
        for _, amount in schedule:
            assert amount == pytest.approx(25.0)

    def test_schedule_is_ascending(self) -> None:
        bond = make_bond(coupon_rate=0.05, coupon_frequency=2, maturity=5.0)
        schedule = coupon_schedule(bond)
        times = [t for t, _ in schedule]
        assert times == sorted(times)

    def test_schedule_ends_at_maturity(self) -> None:
        bond = make_bond(coupon_rate=0.05, coupon_frequency=2, maturity=5.0)
        schedule = coupon_schedule(bond)
        assert schedule[-1][0] == pytest.approx(5.0)

    def test_annual_bond(self) -> None:
        bond = make_bond(coupon_rate=0.04, coupon_frequency=1, maturity=3.0)
        schedule = coupon_schedule(bond)
        assert [t for t, _ in schedule] == pytest.approx([1.0, 2.0, 3.0])
        assert [a for _, a in schedule] == pytest.approx([40.0, 40.0, 40.0])

    def test_zero_rate_zeroes_amounts_but_keeps_dates(self) -> None:
        bond = make_bond(coupon_rate=0.0, coupon_frequency=2, maturity=5.0)
        schedule = coupon_schedule(bond)
        assert len(schedule) == 10
        assert all(a == 0.0 for _, a in schedule)


class TestPriceWithCoupons:
    """Step 5a: discrete coupons injected into the cash-only component."""

    def test_zero_conversion_matches_closed_form_coupon_bond(self) -> None:
        # A coupon bond with no conversion option — the pricer should
        # reproduce the plain credit-risky present value exactly, up to
        # floating-point roundoff. Coupon dates land on tree steps
        # cleanly for these parameters, so there's no timing error.
        face = 1000.0
        rate = 0.05
        freq = 2
        maturity = 5.0
        bond = make_bond(
            face_value=face,
            coupon_rate=rate,
            coupon_frequency=freq,
            maturity=maturity,
            conversion_ratio=0.0,
        )
        market = MarketData(
            spot=100.0,
            volatility=0.3,
            risk_free_rate=0.04,
            credit_spread=0.02,
            dividend_yield=0.02,
        )
        y = market.risk_free_rate + market.credit_spread
        coupon_amount = face * rate / freq
        coupon_times = [(i + 1) * (1.0 / freq) for i in range(int(maturity * freq))]
        expected = sum(coupon_amount * np.exp(-y * t) for t in coupon_times)
        expected += face * np.exp(-y * maturity)
        assert price(bond, market, steps=500).price == pytest.approx(expected, rel=1e-6)

    def test_zero_coupon_matches_pre_coupon_pricer(self) -> None:
        # With coupon_rate = 0, coupon injection contributes zero at
        # every step, so the pricer should behave exactly like step 4.
        # Use the step-4 deep-OTM sanity: price = redemption × exp(-(r+s)T).
        bond = make_bond(
            face_value=1000.0,
            coupon_rate=0.0,
            conversion_ratio=10.0,
            maturity=5.0,
        )
        market = MarketData(
            spot=0.01,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.03,
            dividend_yield=0.0,
        )
        expected = 1000.0 * np.exp(-(0.05 + 0.03) * 5.0)
        assert price(bond, market, steps=500).price == pytest.approx(expected, rel=1e-4)

    def test_deep_itm_with_coupons_still_converts_early(self) -> None:
        # A dividend of 2% on a $200k parity position bleeds ~$7.8k
        # over 2 years — coupons at 5% semi are worth ~$90 PV, which
        # doesn't come close to compensating. Early conversion wins,
        # and every dollar of value ends up in the equity component.
        bond = make_bond(
            face_value=1000.0,
            coupon_rate=0.05,
            coupon_frequency=2,
            maturity=2.0,
            conversion_ratio=10.0,
        )
        market = MarketData(
            spot=10_000.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.02,
        )
        result = price(bond, market, steps=500)
        expected = 10.0 * 10_000.0
        assert result.price == pytest.approx(expected, rel=1e-3)
        assert result.equity_component == pytest.approx(expected, rel=1e-3)
        assert result.cash_only_component == pytest.approx(0.0, abs=expected * 1e-3)

    def test_adding_coupons_increases_price(self) -> None:
        # Compare two otherwise-identical bonds: coupon vs. zero-coupon.
        # A coupon bond is worth strictly more (extra cash flows).
        market = MarketData(
            spot=100.0,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.02,
            dividend_yield=0.02,
        )
        zc = make_bond(coupon_rate=0.0, maturity=5.0, conversion_ratio=10.0)
        cb = make_bond(coupon_rate=0.05, coupon_frequency=2, maturity=5.0, conversion_ratio=10.0)
        assert price(cb, market, steps=500).price > price(zc, market, steps=500).price


class TestPriceWithEarlyExercise:
    """Step 5b: interior put / voluntary conversion / issuer call."""

    _HYBRID_MARKET = MarketData(
        spot=100.0,
        volatility=0.3,
        risk_free_rate=0.05,
        credit_spread=0.02,
        dividend_yield=0.02,
    )

    def _hybrid_bond(self, **overrides) -> ConvertibleBond:
        return make_bond(
            face_value=1000.0,
            coupon_rate=0.05,
            coupon_frequency=2,
            maturity=5.0,
            conversion_ratio=10.0,
            **overrides,
        )

    def test_zero_price_put_is_a_no_op(self) -> None:
        # A put at price 0 is never worth exercising (V_cont > 0).
        bond_no_put = self._hybrid_bond()
        bond_with_put = self._hybrid_bond(put_schedule={0.0: 0.0})
        p_no = price(bond_no_put, self._HYBRID_MARKET, steps=500).price
        p_yes = price(bond_with_put, self._HYBRID_MARKET, steps=500).price
        assert p_yes == pytest.approx(p_no, rel=1e-9)

    def test_huge_call_price_is_a_no_op(self) -> None:
        # A call at a price above any reachable continuation value is
        # never worth exercising (issuer only calls to cap V_cont).
        bond_no_call = self._hybrid_bond()
        bond_with_call = self._hybrid_bond(call_schedule={0.0: 1e12})
        p_no = price(bond_no_call, self._HYBRID_MARKET, steps=500).price
        p_yes = price(bond_with_call, self._HYBRID_MARKET, steps=500).price
        assert p_yes == pytest.approx(p_no, rel=1e-9)

    def test_put_pins_value_when_bond_would_otherwise_be_worthless(self) -> None:
        # Deep OTM with s > 0: un-putted V ≈ redemption × exp(-(r+s)T)
        # ≈ 670, far below the put price P = 1500. The put re-floors
        # V to 1500 at every interior step, and at t=0 the holder puts
        # → V(0) = P exactly, all cash-only.
        put_price_value = 1500.0
        bond = make_bond(
            face_value=1000.0,
            coupon_rate=0.0,
            maturity=5.0,
            conversion_ratio=10.0,
            put_schedule={0.0: put_price_value},
        )
        market = MarketData(
            spot=0.01,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.03,
            dividend_yield=0.0,
        )
        result = price(bond, market, steps=500)
        assert result.price == pytest.approx(put_price_value, rel=1e-6)
        assert result.cash_only_component == pytest.approx(put_price_value, rel=1e-6)
        assert result.equity_component == pytest.approx(0.0, abs=1e-6)

    def test_call_defers_and_prices_at_discounted_strike(self) -> None:
        # Deep OTM bond with call at K = 500 available at every step.
        # Un-called V would be ~redemption × exp(-r T) ≈ 779, which is
        # above K → issuer calls. But the issuer minimizes bond value,
        # so they DEFER the call to the last possible interior step
        # (paying 500 later is cheaper in PV than paying 500 today).
        # The effective payoff is K received at step N-1, so
        # V(0) ≈ K × exp(-r T). This is a subtle but real feature of
        # continuously-callable bonds and worth having as a fixture.
        bond = make_bond(
            face_value=1000.0,
            coupon_rate=0.0,
            maturity=5.0,
            conversion_ratio=1.0,
            call_schedule={0.0: 500.0},
        )
        market = MarketData(
            spot=0.01,
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.0,
        )
        result = price(bond, market, steps=500)
        expected = 500.0 * np.exp(-0.05 * 5.0)
        assert result.price == pytest.approx(expected, rel=1e-2)
        assert result.cash_only_component == pytest.approx(result.price, rel=1e-6)
        assert result.equity_component == pytest.approx(0.0, abs=1.0)

    def test_immediate_call_forces_conversion_when_parity_above_strike(self) -> None:
        # Call at t=0 with K < parity → holder responds by converting.
        # Value = parity, all in the equity component.
        bond = make_bond(
            face_value=1000.0,
            coupon_rate=0.0,
            maturity=5.0,
            conversion_ratio=10.0,
            call_schedule={0.0: 500.0},
        )
        market = MarketData(
            spot=200.0,  # parity = 2000, above K = 500
            volatility=0.3,
            risk_free_rate=0.05,
            credit_spread=0.0,
            dividend_yield=0.0,
        )
        result = price(bond, market, steps=500)
        parity_today = 2000.0
        assert result.price == pytest.approx(parity_today, rel=1e-3)
        assert result.equity_component == pytest.approx(parity_today, rel=1e-3)
        assert result.cash_only_component == pytest.approx(0.0, abs=parity_today * 1e-3)

    def test_adding_put_never_decreases_price(self) -> None:
        # A put is a right for the holder — worth ≥ 0. Attach a put at
        # a plausible price and compare.
        bond_no_put = self._hybrid_bond()
        bond_with_put = self._hybrid_bond(put_schedule={2.0: 950.0, 3.0: 970.0})
        p_no = price(bond_no_put, self._HYBRID_MARKET, steps=500).price
        p_yes = price(bond_with_put, self._HYBRID_MARKET, steps=500).price
        assert p_yes >= p_no - 1e-9

    def test_adding_call_never_increases_price(self) -> None:
        # A call is a right for the issuer — worth ≤ 0 to the holder.
        bond_no_call = self._hybrid_bond()
        bond_with_call = self._hybrid_bond(call_schedule={2.0: 1100.0, 3.0: 1050.0})
        p_no = price(bond_no_call, self._HYBRID_MARKET, steps=500).price
        p_yes = price(bond_with_call, self._HYBRID_MARKET, steps=500).price
        assert p_yes <= p_no + 1e-9
