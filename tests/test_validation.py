"""Tests for the statistical validation framework."""

from __future__ import annotations

import numpy as np
import pytest

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
from validation import (
    DEFAULT_SUITE,
    ValidationResult,
    mean_relative_error,
    print_results,
    run_suite,
    sample_mean,
    sample_variance,
    theoretical_mean,
    theoretical_variance,
    variance_relative_error,
)

# Sample size and tolerances mirror tests/test_distributions.py / conftest.py
# so the framework's per-distribution tests stay numerically consistent with
# the existing test suite.
SAMPLE_SIZE = 50_000
MEAN_RTOL = 0.05
VAR_RTOL = 0.10


def _assert_result_in_tolerance(
    result: ValidationResult,
    expected_mean: float,
    expected_variance: float,
) -> None:
    """Assert a single ``ValidationResult`` matches closed-form moments."""
    assert result.theoretical_mean == pytest.approx(expected_mean)
    assert result.theoretical_variance == pytest.approx(expected_variance)
    assert result.mean_relative_error < MEAN_RTOL, (
        f"mean relative error {result.mean_relative_error:.4%} exceeds {MEAN_RTOL:.0%}"
    )
    assert result.variance_relative_error < VAR_RTOL, (
        f"variance relative error {result.variance_relative_error:.4%} "
        f"exceeds {VAR_RTOL:.0%}"
    )


class TestTheoreticalMetrics:
    """Tests for the pure-function math module."""

    def test_sample_mean_matches_numpy(self) -> None:
        """sample_mean agrees with np.mean."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert sample_mean(samples) == pytest.approx(3.0)

    def test_sample_variance_uses_ddof_zero(self) -> None:
        """sample_variance matches np.var with ddof=0 (population variance)."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Population variance: mean=3, sum((x-3)^2)/5 = 10/5 = 2
        assert sample_variance(samples) == pytest.approx(2.0)

    def test_theoretical_mean_and_variance(self) -> None:
        """theoretical_* delegate to the distribution's methods."""
        dist = ExponentialDistribution(rate=2.0)
        assert theoretical_mean(dist) == pytest.approx(0.5)
        assert theoretical_variance(dist) == pytest.approx(0.25)

    def test_relative_error_zero_when_equal(self) -> None:
        """Relative error is 0 when theoretical equals sample."""
        assert mean_relative_error(1.0, 1.0) == 0.0
        assert variance_relative_error(2.0, 2.0) == 0.0

    def test_relative_error_handles_zero_theoretical(self) -> None:
        """Relative error does not divide by zero."""
        # When theoretical is 0 and sample is also 0, error is 0
        assert mean_relative_error(0.0, 0.0) == 0.0
        # When theoretical is 0 and sample is nonzero, error is inf
        assert mean_relative_error(0.0, 1.0) == float("inf")


class TestRunSuitePerDistribution:
    """Per-distribution tests for the runner.

    One test per distribution in :data:`validation.DEFAULT_SUITE`. Each
    test pins the closed-form theoretical moments and confirms the
    framework reports a sample mean and variance within the standard
    tolerances. Failures point directly at a single distribution.
    """

    def test_exponential(self) -> None:
        """ExponentialDistribution(rate=3.0) — mean 1/3, var 1/9."""
        rate = 3.0
        results = run_suite(
            [ExponentialDistribution(rate=rate)],
            sample_size=SAMPLE_SIZE,
            seed=42,
        )
        _assert_result_in_tolerance(results[0], 1.0 / rate, 1.0 / (rate**2))

    def test_normal(self) -> None:
        """NormalDistribution(mean=10, std=2.5) — mean mu, var sigma^2."""
        mu, sigma = 10.0, 2.5
        results = run_suite(
            [NormalDistribution(mean=mu, std=sigma)],
            sample_size=SAMPLE_SIZE,
            seed=42,
        )
        _assert_result_in_tolerance(results[0], mu, sigma**2)

    def test_gamma(self) -> None:
        """GammaDistribution(shape=2, scale=3) — mean k*theta, var k*theta^2."""
        k, theta = 2.0, 3.0
        results = run_suite(
            [GammaDistribution(shape=k, scale=theta)],
            sample_size=SAMPLE_SIZE,
            seed=42,
        )
        _assert_result_in_tolerance(results[0], k * theta, k * theta**2)

    def test_bernoulli(self) -> None:
        """BernoulliDistribution(p=0.3) — mean p, var p(1-p)."""
        p = 0.3
        results = run_suite(
            [BernoulliDistribution(probability=p)],
            sample_size=SAMPLE_SIZE,
            seed=42,
        )
        _assert_result_in_tolerance(results[0], p, p * (1.0 - p))

    def test_geometric(self) -> None:
        """GeometricDistribution(p=0.4) — mean 1/p, var (1-p)/p^2."""
        p = 0.4
        results = run_suite(
            [GeometricDistribution(probability=p)],
            sample_size=SAMPLE_SIZE,
            seed=42,
        )
        _assert_result_in_tolerance(results[0], 1.0 / p, (1.0 - p) / (p**2))

    def test_weighted_discrete(self) -> None:
        """WeightedDiscrete over {1, 2, 5} with weights {1, 2, 1}.

        Probabilities normalize to 1/4, 1/2, 1/4, giving:
            mean   = 1*(1/4) + 2*(1/2) + 5*(1/4) = 2.5
            var    = 1^2*(1/4) + 2^2*(1/2) + 5^2*(1/4) - mean^2
                   = 0.25 + 2 + 6.25 - 6.25 = 2.25
        """
        values = [1.0, 2.0, 5.0]
        weights = [1.0, 2.0, 1.0]
        results = run_suite(
            [WeightedDiscreteDistribution(values=values, weights=weights)],
            sample_size=SAMPLE_SIZE,
            seed=42,
        )
        _assert_result_in_tolerance(results[0], 2.5, 2.25)


class TestRunSuite:
    """Tests for the runner's suite-level behavior."""

    def test_run_suite_returns_one_result_per_distribution(self) -> None:
        """The suite yields one ValidationResult per input distribution."""
        results = run_suite(DEFAULT_SUITE, sample_size=1_000, seed=42)
        assert len(results) == len(DEFAULT_SUITE)
        assert all(isinstance(r, ValidationResult) for r in results)
        assert all(r.name for r in results)

    def test_run_suite_is_reproducible_with_same_seed(self) -> None:
        """Two runs with the same seed produce identical results."""
        a = run_suite(DEFAULT_SUITE, sample_size=5_000, seed=123)
        b = run_suite(DEFAULT_SUITE, sample_size=5_000, seed=123)
        assert a == b


class TestPrintResults:
    """Tests for the table-printing helper."""

    def test_print_results_includes_all_distribution_names(self, capsys) -> None:
        """The printed table contains the name of every distribution."""
        results = run_suite(DEFAULT_SUITE, sample_size=1_000, seed=42)
        print_results(results)
        captured = capsys.readouterr().out

        # Header is present
        assert "Distribution" in captured
        # Every distribution name appears in the output
        for result in results:
            assert result.name in captured
