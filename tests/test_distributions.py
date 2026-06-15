"""Statistical validation tests for probability distributions."""

import numpy as np
import pytest

from conftest import assert_moments_close
from distributions.continuous import (
    ExponentialDistribution,
    GammaDistribution,
    NormalDistribution,
)
from distributions.discrete import (
    BernoulliDistribution,
    GeometricDistribution,
    WeightedDiscreteDistribution,
)
from distributions.poisson import HomogeneousPoissonProcess

SAMPLE_SIZE = 50_000
SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a seeded random generator."""
    return np.random.default_rng(SEED)


class TestExponentialDistribution:
    """Tests for ExponentialDistribution."""

    def test_moments(self, rng: np.random.Generator) -> None:
        """Sample mean and variance match theory."""
        rate = 3.0
        dist = ExponentialDistribution(rate=rate, rng=rng)
        samples = dist.sample(SAMPLE_SIZE)
        assert_moments_close(samples, dist.mean(), dist.variance())

    def test_invalid_rate(self) -> None:
        """Non-positive rate is rejected."""
        with pytest.raises(ValueError):
            ExponentialDistribution(rate=0.0)


class TestNormalDistribution:
    """Tests for NormalDistribution."""

    def test_moments(self, rng: np.random.Generator) -> None:
        """Sample mean and variance match theory."""
        dist = NormalDistribution(mean=10.0, std=2.5, rng=rng)
        samples = dist.sample(SAMPLE_SIZE)
        assert_moments_close(samples, dist.mean(), dist.variance())


class TestGammaDistribution:
    """Tests for GammaDistribution."""

    def test_moments(self, rng: np.random.Generator) -> None:
        """Sample mean and variance match theory."""
        dist = GammaDistribution(shape=2.0, scale=3.0, rng=rng)
        samples = dist.sample(SAMPLE_SIZE)
        assert_moments_close(samples, dist.mean(), dist.variance())


class TestBernoulliDistribution:
    """Tests for BernoulliDistribution."""

    def test_moments(self, rng: np.random.Generator) -> None:
        """Sample mean and variance match theory."""
        dist = BernoulliDistribution(probability=0.3, rng=rng)
        samples = dist.sample(SAMPLE_SIZE)
        assert_moments_close(samples, dist.mean(), dist.variance())

    def test_invalid_probability(self) -> None:
        """Boundary probabilities are rejected."""
        with pytest.raises(ValueError):
            BernoulliDistribution(probability=0.0)
        with pytest.raises(ValueError):
            BernoulliDistribution(probability=1.0)


class TestGeometricDistribution:
    """Tests for GeometricDistribution."""

    def test_moments(self, rng: np.random.Generator) -> None:
        """Sample mean and variance match theory."""
        dist = GeometricDistribution(probability=0.4, rng=rng)
        samples = dist.sample(SAMPLE_SIZE)
        assert_moments_close(samples, dist.mean(), dist.variance())


class TestWeightedDiscreteDistribution:
    """Tests for WeightedDiscreteDistribution."""

    def test_moments(self, rng: np.random.Generator) -> None:
        """Sample mean and variance match theory."""
        dist = WeightedDiscreteDistribution(
            values=[1.0, 2.0, 5.0],
            weights=[1.0, 2.0, 1.0],
            rng=rng,
        )
        samples = dist.sample(SAMPLE_SIZE)
        assert_moments_close(samples, dist.mean(), dist.variance())

    def test_mismatched_lengths(self) -> None:
        """Mismatched values/weights raise ValueError."""
        with pytest.raises(ValueError):
            WeightedDiscreteDistribution(values=[1.0], weights=[1.0, 2.0])


class TestHomogeneousPoissonProcess:
    """Tests for HomogeneousPoissonProcess."""

    def test_inter_arrival_is_exponential(self, rng: np.random.Generator) -> None:
        """Inter-arrival times follow exponential distribution."""
        rate = 5.0
        process = HomogeneousPoissonProcess(arrival_rate=rate, rng=rng)
        samples = np.array([process.sample_inter_arrival_time() for _ in range(SAMPLE_SIZE)])
        expected_mean = 1.0 / rate
        expected_var = 1.0 / (rate**2)
        assert_moments_close(samples, expected_mean, expected_var)

    def test_generate_arrival_times_ordered(self, rng: np.random.Generator) -> None:
        """Generated arrival times are sorted within the window."""
        process = HomogeneousPoissonProcess(arrival_rate=10.0, rng=rng)
        arrivals = process.generate_arrival_times(start_time=0.0, end_time=10.0)
        assert arrivals == sorted(arrivals)
        assert all(0.0 <= t < 10.0 for t in arrivals)

    def test_expected_arrivals(self) -> None:
        """Expected arrival count equals lambda * duration."""
        process = HomogeneousPoissonProcess(arrival_rate=4.0)
        assert process.expected_arrivals(25.0) == 100.0

    def test_invalid_time_window(self, rng: np.random.Generator) -> None:
        """Invalid time window raises ValueError."""
        process = HomogeneousPoissonProcess(arrival_rate=1.0, rng=rng)
        with pytest.raises(ValueError):
            process.generate_arrival_times(start_time=10.0, end_time=5.0)
