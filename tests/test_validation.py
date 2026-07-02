"""Tests for the statistical validation framework."""

from __future__ import annotations

import numpy as np
import pytest

from distributions.continuous import ExponentialDistribution
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


class TestRunSuite:
    """Tests for the runner."""

    def test_run_suite_returns_one_result_per_distribution(self) -> None:
        """The suite yields one ValidationResult per input distribution."""
        results = run_suite(DEFAULT_SUITE, sample_size=1_000, seed=42)
        assert len(results) == len(DEFAULT_SUITE)
        assert all(isinstance(r, ValidationResult) for r in results)
        assert all(r.name for r in results)

    def test_exponential_meets_tolerances(self) -> None:
        """Exponential at rate=3.0 passes the standard 5% / 10% tolerances."""
        dist = ExponentialDistribution(rate=3.0)
        results = run_suite([dist], sample_size=50_000, seed=42)
        assert len(results) == 1
        r = results[0]
        assert r.theoretical_mean == pytest.approx(1.0 / 3.0)
        assert r.theoretical_variance == pytest.approx(1.0 / 9.0)
        assert r.mean_relative_error < 0.05
        assert r.variance_relative_error < 0.10

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
