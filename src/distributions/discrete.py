"""Discrete probability distributions."""

import numpy as np

from distributions.base import Distribution


class BernoulliDistribution(Distribution):
    """Bernoulli distribution with success probability p."""

    def __init__(self, probability: float, rng: np.random.Generator | None = None) -> None:
        """Initialize a Bernoulli distribution.

        Parameters:
            probability: Success probability in [0, 1].
            rng: Optional NumPy random generator.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        super().__init__(rng)
        self._probability = probability

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw Bernoulli random variates (0 or 1).

        Parameters:
            size: Number of samples.

        Returns:
            Array of Bernoulli samples.
        """
        return self._rng.binomial(n=1, p=self._probability, size=size).astype(float)

    def mean(self) -> float:
        """Return theoretical mean p.

        Returns:
            Expected value.
        """
        return self._probability

    def variance(self) -> float:
        """Return theoretical variance p(1-p).

        Returns:
            Variance.
        """
        return self._probability * (1.0 - self._probability)


class GeometricDistribution(Distribution):
    """Geometric distribution counting trials until first success.

    Parameterized by success probability p. Support: {1, 2, 3, ...}.
    Mean = 1/p, Variance = (1-p)/p^2.
    """

    def __init__(self, probability: float, rng: np.random.Generator | None = None) -> None:
        """Initialize a geometric distribution.

        Parameters:
            probability: Success probability in (0, 1].
            rng: Optional NumPy random generator.
        """
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        super().__init__(rng)
        self._probability = probability

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw geometric random variates.

        Parameters:
            size: Number of samples.

        Returns:
            Array of geometric samples (integers >= 1).
        """
        return self._rng.geometric(p=self._probability, size=size).astype(float)

    def mean(self) -> float:
        """Return theoretical mean 1/p.

        Returns:
            Expected value.
        """
        return 1.0 / self._probability

    def variance(self) -> float:
        """Return theoretical variance (1-p)/p^2.

        Returns:
            Variance.
        """
        p = self._probability
        return (1.0 - p) / (p**2)


class WeightedDiscreteDistribution(Distribution):
    """Discrete distribution over a finite set of values with weights."""

    def __init__(
        self,
        values: list[float],
        weights: list[float],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize a weighted discrete distribution.

        Parameters:
            values: Support of the distribution.
            weights: Non-negative weights (need not sum to 1).
            rng: Optional NumPy random generator.
        """
        if len(values) == 0:
            raise ValueError("values must not be empty")
        if len(values) != len(weights):
            raise ValueError("values and weights must have equal length")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        if sum(weights) == 0:
            raise ValueError("sum of weights must be positive")

        super().__init__(rng)
        self._values = np.array(values, dtype=float)
        self._weights = np.array(weights, dtype=float)
        self._probabilities = self._weights / self._weights.sum()

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw weighted discrete random variates.

        Parameters:
            size: Number of samples.

        Returns:
            Array of sampled values from the support.
        """
        return self._rng.choice(self._values, size=size, p=self._probabilities)

    def mean(self) -> float:
        """Return theoretical mean sum(v_i * p_i).

        Returns:
            Expected value.
        """
        return float(np.dot(self._values, self._probabilities))

    def variance(self) -> float:
        """Return theoretical variance.

        Returns:
            Variance.
        """
        mean = self.mean()
        return float(np.dot(self._probabilities, (self._values - mean) ** 2))
