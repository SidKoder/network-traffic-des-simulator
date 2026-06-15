"""Continuous probability distributions."""

import numpy as np

from distributions.base import Distribution


class ExponentialDistribution(Distribution):
    """Exponential distribution parameterized by rate.

    Parameters:
        rate: Rate parameter (lambda > 0). Mean = 1/rate.
    """

    def __init__(self, rate: float, rng: np.random.Generator | None = None) -> None:
        """Initialize an exponential distribution.

        Parameters:
            rate: Positive rate parameter.
            rng: Optional NumPy random generator.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        super().__init__(rng)
        self._rate = rate

    @property
    def rate(self) -> float:
        """Return the rate parameter.

        Returns:
            Rate lambda.
        """
        return self._rate

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw exponential random variates.

        Parameters:
            size: Number of samples.

        Returns:
            Array of exponential samples.
        """
        return self._rng.exponential(scale=1.0 / self._rate, size=size)

    def mean(self) -> float:
        """Return theoretical mean 1/rate.

        Returns:
            Expected value.
        """
        return 1.0 / self._rate

    def variance(self) -> float:
        """Return theoretical variance 1/rate^2.

        Returns:
            Variance.
        """
        return 1.0 / (self._rate**2)


class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution.

    Parameters:
        mean: Location parameter (mu).
        std: Scale parameter (sigma > 0).
    """

    def __init__(
        self,
        mean: float,
        std: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize a normal distribution.

        Parameters:
            mean: Mean of the distribution.
            std: Standard deviation (must be positive).
            rng: Optional NumPy random generator.
        """
        if std <= 0:
            raise ValueError("std must be positive")
        super().__init__(rng)
        self._mean = mean
        self._std = std

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw normal random variates.

        Parameters:
            size: Number of samples.

        Returns:
            Array of normal samples.
        """
        return self._rng.normal(loc=self._mean, scale=self._std, size=size)

    def mean(self) -> float:
        """Return theoretical mean mu.

        Returns:
            Expected value.
        """
        return self._mean

    def variance(self) -> float:
        """Return theoretical variance sigma^2.

        Returns:
            Variance.
        """
        return self._std**2


class GammaDistribution(Distribution):
    """Gamma distribution with shape and scale parameters.

    Uses NumPy's parameterization: shape (k) and scale (theta).
    Mean = k * theta, Variance = k * theta^2.
    """

    def __init__(
        self,
        shape: float,
        scale: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize a gamma distribution.

        Parameters:
            shape: Shape parameter k (must be positive).
            scale: Scale parameter theta (must be positive).
            rng: Optional NumPy random generator.
        """
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be positive")
        super().__init__(rng)
        self._shape = shape
        self._scale = scale

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw gamma random variates.

        Parameters:
            size: Number of samples.

        Returns:
            Array of gamma samples.
        """
        return self._rng.gamma(shape=self._shape, scale=self._scale, size=size)

    def mean(self) -> float:
        """Return theoretical mean k * theta.

        Returns:
            Expected value.
        """
        return self._shape * self._scale

    def variance(self) -> float:
        """Return theoretical variance k * theta^2.

        Returns:
            Variance.
        """
        return self._shape * self._scale**2
