"""Tests for JTD space and time grids."""

from math import exp, sqrt

import numpy as np
import pytest

from tf_convert_pricer.jtd.grid import (
    GridSpec,
    node_index,
    smax_for,
    space_grid,
    time_grid,
)


class TestSpaceGrid:
    @pytest.mark.parametrize("concentration", [None, 0.3])
    def test_spot_exactly_on_node(self, concentration: float | None) -> None:
        spots, j0 = space_grid(
            37.3, 900.0, GridSpec(space_nodes=300, concentration=concentration)
        )
        assert spots[j0] == 37.3
        assert spots[0] == 0.0
        assert np.all(np.diff(spots) > 0)
        assert spots[-1] >= 900.0 - 1e-9

    def test_sinh_grid_is_denser_near_spot(self) -> None:
        spots, j0 = space_grid(
            100.0, 1000.0, GridSpec(space_nodes=400, concentration=0.3)
        )
        h = np.diff(spots)
        assert h[j0] < h[-1] / 5

    def test_pins_land_on_nodes(self) -> None:
        spots, j0 = space_grid(
            100.0, 1000.0, GridSpec(space_nodes=400), pins=(130.0, 83.3)
        )
        assert 130.0 in spots and 83.3 in spots and spots[j0] == 100.0
        assert np.all(np.diff(spots) > 0)

    def test_smax_rule(self) -> None:
        smax = smax_for(100.0, 125.0, 0.25, 5.0, GridSpec())
        assert smax == pytest.approx(4 * 125.0 * exp(3 * 0.25 * sqrt(5.0)))


class TestTimeGrid:
    def test_events_are_nodes_and_steps_bounded(self) -> None:
        events = [0.5, 1.0, 1.37, 3.0]
        times = time_grid(5.0, events, steps_per_year=50)
        assert times[0] == 0.0 and times[-1] == pytest.approx(5.0)
        assert all(node_index(times, t) is not None for t in events)
        assert np.max(np.diff(times)) <= 1 / 50 + 1e-12

    def test_ignores_events_outside_life(self) -> None:
        times = time_grid(1.0, [-0.5, 0.0, 2.0], steps_per_year=10)
        assert len(times) == 11
