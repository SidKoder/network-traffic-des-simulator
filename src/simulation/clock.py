"""Simulation clock for discrete event time advancement."""

class SimulationClock:
    """Maintains and advances simulation time.

    Time advances only when explicitly moved forward via event timestamps.
    The clock never uses real-time or sleep-based progression.
    """

    def __init__(self, initial_time: float = 0.0) -> None:
        """Initialize the simulation clock.

        Parameters:
            initial_time: Starting simulation time (must be non-negative).
        """
        if initial_time < 0:
            raise ValueError("initial_time must be non-negative")
        self._current_time = initial_time

    @property
    def current_time(self) -> float:
        """Return the current simulation time.

        Returns:
            Current time value.
        """
        return self._current_time

    def advance_to(self, timestamp: float) -> float:
        """Advance the clock to a new timestamp.

        Parameters:
            timestamp: Target simulation time (must not precede current time).

        Returns:
            The new current time after advancement.

        Raises:
            ValueError: If timestamp is earlier than current time.
        """
        if timestamp < self._current_time:
            raise ValueError(
                f"Cannot move clock backward: {timestamp} < {self._current_time}"
            )
        self._current_time = timestamp
        return self._current_time

    def reset(self, initial_time: float = 0.0) -> None:
        """Reset the clock to a new starting time.

        Parameters:
            initial_time: Time to reset to (must be non-negative).
        """
        if initial_time < 0:
            raise ValueError("initial_time must be non-negative")
        self._current_time = initial_time
