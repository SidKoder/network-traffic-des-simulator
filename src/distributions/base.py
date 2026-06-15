"""Abstract base for probability distributions."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator


class Distribution(ABC):
    """Base class for all probability distributions.

    Distributions are network-agnostic and depend only on NumPy for sampling.
    """

    def __init__(self, rng: Generator | None = None) -> None:
        """Initialize the distribution with an optional RNG.

        Parameters:
            rng: NumPy random generator. A default generator is created if None.
        """
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def rng(self) -> Generator:
        """Return the underlying random number generator.

        Returns:
            NumPy Generator instance.
        """
        return self._rng

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        """Draw random samples from the distribution.

        Parameters:
            size: Number of samples to draw.

        Returns:
            Array of sampled values.
        """

    @abstractmethod
    def mean(self) -> float:
        """Return the theoretical mean of the distribution.

        Returns:
            Expected value.
        """

    @abstractmethod
    def variance(self) -> float:
        """Return the theoretical variance of the distribution.

        Returns:
            Variance.
        """

    def std(self) -> float:
        """Return the theoretical standard deviation.

        Returns:
            Square root of variance.
        """
        return float(np.sqrt(self.variance()))
