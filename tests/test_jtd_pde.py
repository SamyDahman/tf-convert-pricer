"""Tests for the generic θ-scheme PDE core."""

from math import exp

import numpy as np
import pytest

from tf_convert_pricer.jtd.grid import GridSpec, space_grid
from tf_convert_pricer.jtd.pde import operator_bands, solve_backward


@pytest.fixture
def spots() -> np.ndarray:
    return space_grid(100.0, 800.0, GridSpec(space_nodes=200))[0]


def _apply(bands, v):
    lower, diag, upper = bands
    out = diag * v
    out[1:] += lower[1:] * v[:-1]
    out[:-1] += upper[:-1] * v[1:]
    return out


class TestOperator:
    def test_annihilates_constants_without_kill(self, spots) -> None:
        bands = operator_bands(
            spots, 0.3, np.full_like(spots, 0.05), np.zeros_like(spots)
        )
        assert np.allclose(_apply(bands, np.ones_like(spots)), 0.0)

    def test_linear_function_gives_drift(self, spots) -> None:
        # L S = μ S exactly (V_SS = 0) — including the Smax boundary row.
        mu = np.full_like(spots, 0.07)
        bands = operator_bands(spots, 0.3, mu, np.zeros_like(spots))
        assert np.allclose(_apply(bands, spots)[1:], (mu * spots)[1:])

    def test_off_diagonals_nonnegative_with_large_drift(self, spots) -> None:
        # ηλS drift is huge near S = 0 when λ(S) blows up; upwinding
        # keeps the scheme monotone.
        mu = 2.0 * (100.0 / np.maximum(spots, 1e-9))
        lower, _, upper = operator_bands(spots, 0.2, mu, np.zeros_like(spots))
        assert (lower[1:-1] >= 0).all() and (upper[1:-1] >= 0).all()


class TestSolveBackward:
    def test_pure_discounting(self, spots) -> None:
        times = np.linspace(0.0, 2.0, 201)
        kill = np.full_like(spots, 0.05)

        def coefficients(t):
            return np.full_like(spots, 0.05), kill, np.zeros_like(spots)

        values = solve_backward(spots, times, 0.3, np.ones_like(spots), coefficients)
        assert values == pytest.approx(np.full_like(spots, exp(-0.1)), abs=1e-6)

    def test_multiple_right_hand_sides(self, spots) -> None:
        # Annuity leg (source 1) alongside a zero-source column.
        times = np.linspace(0.0, 1.0, 101)

        def coefficients(t):
            source = np.column_stack((np.zeros_like(spots), np.ones_like(spots)))
            return np.zeros_like(spots), np.full_like(spots, 0.05), source

        values = solve_backward(
            spots, times, 0.3, np.zeros((len(spots), 2)), coefficients
        )
        assert values[:, 0] == pytest.approx(0.0)
        # Rannacher start-up steps are first order: O(Δt²)-sized, not exact.
        assert values[:, 1] == pytest.approx((1 - exp(-0.05)) / 0.05, abs=1e-5)
