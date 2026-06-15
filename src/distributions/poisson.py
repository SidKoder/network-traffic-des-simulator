"""Poisson arrival process for packet generation."""

import numpy as np

from distributions.continuous import ExponentialDistribution


class HomogeneousPoissonProcess:
    """Homogeneous Poisson process with constant arrival rate.

    Inter-arrival times are i.i.d. exponential with rate lambda.
    This class is network-agnostic and produces only timing information.
    """

    def __init__(self, arrival_rate: float, rng: np.random.Generator | None = None) -> None:
        """Initialize a homogeneous Poisson process.

        Parameters:
            arrival_rate: Arrival rate lambda (packets per unit time).
            rng: Optional NumPy random generator.
        """
        self._arrival_rate = arrival_rate
        self._inter_arrival = ExponentialDistribution(rate=arrival_rate, rng=rng)

    @property
    def arrival_rate(self) -> float:
        """Return the Poisson arrival rate.

        Returns:
            Arrival rate lambda.
        """
        return self._arrival_rate

    def sample_inter_arrival_time(self) -> float:
        """Draw a single inter-arrival time.

        Returns:
            Time until the next arrival.
        """
        return float(self._inter_arrival.sample(1)[0])

    def generate_arrival_times(
        self,
        start_time: float,
        end_time: float,
    ) -> list[float]:
        """Generate arrival timestamps within a time window.

        Parameters:
            start_time: Beginning of the observation window.
            end_time: End of the observation window.

        Returns:
            Sorted list of arrival timestamps in [start_time, end_time).
        """
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")

        arrivals: list[float] = []
        current = start_time

        while True:
            current += self.sample_inter_arrival_time()
            if current >= end_time:
                break
            arrivals.append(current)

        return arrivals

    def expected_arrivals(self, duration: float) -> float:
        """Return expected number of arrivals over a duration.

        Parameters:
            duration: Length of the observation interval.

        Returns:
            Expected arrival count lambda * duration.
        """
        return self._arrival_rate * duration
