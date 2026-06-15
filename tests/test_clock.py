"""Tests for simulation clock."""

import pytest

from simulation.clock import SimulationClock


class TestSimulationClock:
    """Tests for SimulationClock time advancement."""

    def test_initial_time(self) -> None:
        """Clock starts at specified initial time."""
        clock = SimulationClock(initial_time=5.0)
        assert clock.current_time == 5.0

    def test_advance_to_future(self) -> None:
        """Clock advances to a later timestamp."""
        clock = SimulationClock()
        new_time = clock.advance_to(10.5)
        assert new_time == 10.5
        assert clock.current_time == 10.5

    def test_advance_to_same_time(self) -> None:
        """Advancing to current time is a no-op."""
        clock = SimulationClock(initial_time=3.0)
        clock.advance_to(3.0)
        assert clock.current_time == 3.0

    def test_cannot_advance_backward(self) -> None:
        """Backward time travel raises ValueError."""
        clock = SimulationClock()
        clock.advance_to(5.0)
        with pytest.raises(ValueError, match="Cannot move clock backward"):
            clock.advance_to(3.0)

    def test_negative_initial_time_rejected(self) -> None:
        """Negative initial time raises ValueError."""
        with pytest.raises(ValueError):
            SimulationClock(initial_time=-1.0)

    def test_reset(self) -> None:
        """Reset returns clock to a new starting point."""
        clock = SimulationClock()
        clock.advance_to(100.0)
        clock.reset(initial_time=0.0)
        assert clock.current_time == 0.0
