"""Tests for the closed forms used as JTD validation anchors."""

from math import exp

import pytest

from tf_convert_pricer.jtd.analytic import black_scholes_call, risky_zero_coupon


def test_black_scholes_textbook_value() -> None:
    # Hull: S=42, K=40, r=10%, σ=20%, T=0.5 → 4.76
    assert black_scholes_call(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(
        4.759, abs=1e-3
    )


def test_black_scholes_degenerate_inputs() -> None:
    assert black_scholes_call(120.0, 100.0, 0.0, 0.05, 0.2) == 20.0
    assert black_scholes_call(0.0, 100.0, 1.0, 0.05, 0.2) == 0.0


def test_risky_zero_reduces_to_riskless_without_default() -> None:
    assert risky_zero_coupon(100.0, 5.0, 0.04, 0.0, 0.4) == pytest.approx(
        100 * exp(-0.2)
    )


def test_spec_t2_closed_form() -> None:
    assert risky_zero_coupon(100.0, 5.0, 0.04, 0.03, 0.4) == pytest.approx(
        75.5313, abs=1e-4
    )
