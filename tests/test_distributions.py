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
        """Probabilities outside [0, 1] are rejected."""
        with pytest.raises(ValueError):
            BernoulliDistribution(probability=-0.1)
        with pytest.raises(ValueError):
            BernoulliDistribution(probability=1.1)

    def test_boundary_probabilities_are_deterministic(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Endpoint probabilities produce deterministic Bernoulli samples."""
        never = BernoulliDistribution(probability=0.0, rng=rng)
        always = BernoulliDistribution(probability=1.0, rng=rng)

        assert np.array_equal(never.sample(20), np.zeros(20))
        assert np.array_equal(always.sample(20), np.ones(20))


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

    def test_invalid_arrival_rate(self) -> None:
        """Non-positive arrival_rate is rejected."""
        with pytest.raises(ValueError):
            HomogeneousPoissonProcess(arrival_rate=0.0)
        with pytest.raises(ValueError):
            HomogeneousPoissonProcess(arrival_rate=-1.0)

    def test_batched_inter_arrival_times(self, rng: np.random.Generator) -> None:
        """Batched method returns n samples matching the exponential distribution."""
        rate = 7.0
        process = HomogeneousPoissonProcess(arrival_rate=rate, rng=rng)
        samples = process.sample_inter_arrival_times(SAMPLE_SIZE)
        assert_moments_close(samples, 1.0 / rate, 1.0 / (rate**2))

        # Edge case: zero samples returns an empty array
        assert process.sample_inter_arrival_times(0).shape == (0,)

        # Edge case: negative n is rejected
        with pytest.raises(ValueError):
            process.sample_inter_arrival_times(-1)

    def test_arrival_count_matches_expected(self, rng: np.random.Generator) -> None:
        """Mean arrival count over many windows equals lambda * duration."""
        rate = 10.0
        duration = 100.0
        n_windows = 200
        process = HomogeneousPoissonProcess(arrival_rate=rate, rng=rng)

        counts = np.array([
            len(process.generate_arrival_times(0.0, duration))
            for _ in range(n_windows)
        ])
        expected_count = rate * duration  # = 1000
        # The sample mean of independent Poisson(1000) counts has
        # std ~ sqrt(1000 / 200) ~ 2.24, so a 5% relative tolerance
        # is ~22 standard errors — extremely loose, but guards
        # against an order-of-magnitude regression in the vectorized
        # implementation.
        assert np.mean(counts) == pytest.approx(expected_count, rel=0.05)
