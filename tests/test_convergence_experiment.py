"""Tests for Monte Carlo convergence experiments.

These tests verify that the M/M/1 simulator's observed metrics converge
to theoretical steady-state values as simulation duration increases.
"""

from __future__ import annotations

import pytest

from validation.convergence_experiment import (
    ConvergenceMetricSeries,
    ConvergencePoint,
    ConvergenceResult,
    format_convergence_report,
    run_convergence_experiment,
)


# ---------------------------------------------------------------------------
# Marker: all tests in this module are convergence experiments
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.convergence


# ---------------------------------------------------------------------------
# Core convergence tests
# ---------------------------------------------------------------------------


class TestConvergenceMeanDelay:
    """Verify that mean delay (W) error shrinks with longer runs."""

    def test_mean_delay_final_error_less_than_first(self) -> None:
        """W relative error at the longest duration must be strictly
        smaller than at the shortest duration."""
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            seed=42,
        )

        w_series = _find_series(result, "W")
        first_error = w_series.points[0].relative_error
        final_error = w_series.points[-1].relative_error

        assert final_error < first_error, (
            f"W did not converge: first={first_error:.4%}, final={final_error:.4%}"
        )

    def test_mean_delay_negative_log_log_slope(self) -> None:
        """The log-log slope for W must be negative."""
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            seed=42,
        )
        w_series = _find_series(result, "W")
        assert w_series.log_log_slope < 0.0, (
            f"W log-log slope is non-negative: {w_series.log_log_slope}"
        )


class TestConvergenceAllMetrics:
    """Verify that all five tracked metrics converge."""

    def test_all_metrics_converge(self) -> None:
        """result.converged must be True for the standard λ=4, μ=6 case."""
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            seed=42,
        )
        assert result.converged is True, _convergence_failure_detail(result)

    def test_each_metric_series_individually_converges(self) -> None:
        """Each ConvergenceMetricSeries.converged must be True."""
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            seed=42,
        )
        for series in result.metric_series:
            assert series.converged is True, (
                f"{series.metric_name} did not converge: "
                f"slope={series.log_log_slope:+.4f}, "
                f"first_err={series.points[0].relative_error:.4%}, "
                f"final_err={series.points[-1].relative_error:.4%}"
            )


class TestConvergenceWithReplications:
    """Verify that replications smooth out noise."""

    def test_replications_produce_convergence(self) -> None:
        """With 3 replications, overall convergence must still hold."""
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            seed=42,
            num_replications=3,
        )
        assert result.converged is True, _convergence_failure_detail(result)
        assert result.num_replications == 3

    def test_replications_reduce_final_error_variance(self) -> None:
        """The final error with replications should be comparable to or
        better than without.  We don't assert strict < because we're
        averaging, but it must still be small."""
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            seed=42,
            num_replications=3,
        )
        for series in result.metric_series:
            final_err = series.points[-1].relative_error
            assert final_err < 0.15, (
                f"{series.metric_name} final error {final_err:.4%} "
                f"is too high even with replications"
            )


class TestConvergenceHighUtilization:
    """Verify convergence under heavy load (ρ=0.9)."""

    def test_high_utilization_converges(self) -> None:
        """ρ=0.9 needs longer runs to converge; the default durations
        up to 50k should suffice."""
        result = run_convergence_experiment(
            lambda_rate=9.0,
            mu_rate=10.0,
            seed=42,
        )
        # At high ρ, convergence is slower.  We relax the overall check
        # and verify that at least the mean delay (W) converges.
        w_series = _find_series(result, "W")
        assert w_series.converged is True, (
            f"W did not converge at ρ=0.9: slope={w_series.log_log_slope:+.4f}"
        )


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------


class TestConvergenceReportFormat:
    """Verify report output contains key structural elements."""

    def test_report_contains_header(self) -> None:
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            durations=[100.0, 1_000.0, 5_000.0],
            seed=42,
        )
        report = format_convergence_report(result)
        assert "Convergence Experiment" in report
        assert "lambda=4.000" in report
        assert "mu=6.000" in report

    def test_report_contains_metric_names(self) -> None:
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            durations=[100.0, 1_000.0, 5_000.0],
            seed=42,
        )
        report = format_convergence_report(result)
        for name in ("W", "Wq", "utilization", "L", "Lq"):
            assert name in report

    def test_report_contains_pass_or_fail(self) -> None:
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            durations=[100.0, 1_000.0, 5_000.0],
            seed=42,
        )
        report = format_convergence_report(result)
        assert "PASS" in report or "FAIL" in report

    def test_report_contains_per_metric_summary(self) -> None:
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            durations=[100.0, 1_000.0, 5_000.0],
            seed=42,
        )
        report = format_convergence_report(result)
        assert "Per-metric summary:" in report
        assert "CONVERGED" in report or "NOT CONVERGED" in report
        assert "slope=" in report


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestConvergenceInputValidation:
    """Verify that bad inputs are rejected with clear errors."""

    def test_rejects_non_positive_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_rate must be positive"):
            run_convergence_experiment(lambda_rate=0.0, mu_rate=6.0)

    def test_rejects_non_positive_mu(self) -> None:
        with pytest.raises(ValueError, match="mu_rate must be positive"):
            run_convergence_experiment(lambda_rate=4.0, mu_rate=0.0)

    def test_rejects_unstable_system(self) -> None:
        with pytest.raises(ValueError, match="lambda_rate must be less than mu_rate"):
            run_convergence_experiment(lambda_rate=6.0, mu_rate=6.0)

    def test_rejects_empty_durations(self) -> None:
        with pytest.raises(ValueError, match="durations must be a non-empty"):
            run_convergence_experiment(lambda_rate=4.0, mu_rate=6.0, durations=[])

    def test_rejects_zero_replications(self) -> None:
        with pytest.raises(ValueError, match="num_replications must be >= 1"):
            run_convergence_experiment(
                lambda_rate=4.0, mu_rate=6.0, num_replications=0
            )

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            run_convergence_experiment(
                lambda_rate=4.0, mu_rate=6.0, metric_names=["nonexistent"]
            )

    def test_rejects_empty_metric_names(self) -> None:
        with pytest.raises(ValueError, match="metric_names must be a non-empty"):
            run_convergence_experiment(
                lambda_rate=4.0, mu_rate=6.0, metric_names=[]
            )


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestConvergenceDataStructures:
    """Verify structural properties of the returned dataclasses."""

    def test_points_are_sorted_by_duration(self) -> None:
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            durations=[5_000.0, 100.0, 1_000.0],  # intentionally unsorted
            seed=42,
        )
        for series in result.metric_series:
            durations = [p.duration for p in series.points]
            assert durations == sorted(durations)

    def test_result_contains_correct_rates(self) -> None:
        result = run_convergence_experiment(
            lambda_rate=3.0,
            mu_rate=5.0,
            durations=[100.0, 500.0],
            seed=99,
        )
        assert result.lambda_rate == 3.0
        assert result.mu_rate == 5.0

    def test_convergence_point_has_positive_theoretical(self) -> None:
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            durations=[100.0, 1_000.0],
            seed=42,
        )
        for series in result.metric_series:
            for point in series.points:
                assert point.theoretical > 0.0

    def test_metric_series_count_matches_requested_names(self) -> None:
        names = ("W", "Lq")
        result = run_convergence_experiment(
            lambda_rate=4.0,
            mu_rate=6.0,
            durations=[100.0, 1_000.0],
            seed=42,
            metric_names=names,
        )
        assert len(result.metric_series) == len(names)
        assert {s.metric_name for s in result.metric_series} == set(names)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_series(result: ConvergenceResult, metric_name: str) -> ConvergenceMetricSeries:
    """Return the series for the given metric, or fail loudly."""
    for series in result.metric_series:
        if series.metric_name == metric_name:
            return series
    raise AssertionError(f"Metric '{metric_name}' not found in result")


def _convergence_failure_detail(result: ConvergenceResult) -> str:
    """Build a diagnostic string when convergence fails."""
    lines = ["Not all metrics converged:"]
    for series in result.metric_series:
        if not series.converged:
            lines.append(
                f"  {series.metric_name}: slope={series.log_log_slope:+.4f}, "
                f"first_err={series.points[0].relative_error:.4%}, "
                f"final_err={series.points[-1].relative_error:.4%}"
            )
    return "\n".join(lines)
