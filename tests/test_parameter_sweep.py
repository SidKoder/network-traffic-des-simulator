"""Tests for parameter sweeps verification."""

from __future__ import annotations

import pytest

from validation.parameter_sweep import (
    ParameterSweepResult,
    SweepPoint,
    format_parameter_sweep_report,
    run_parameter_sweep,
)

# Mark all tests in this module with the 'sweep' marker
pytestmark = pytest.mark.sweep


class TestParameterSweepCore:
    """Verify core functionality of the parameter sweep framework."""

    def test_parameter_sweep_runs_successfully_infinite_capacity(self) -> None:
        """Verify standard stable M/M/1 sweep has reasonable trend direction."""
        lambda_rates = [1.0, 3.0, 5.0, 8.0]
        mu_rate = 10.0
        result = run_parameter_sweep(
            lambda_rates=lambda_rates,
            mu_rate=mu_rate,
            capacity=None,
            simulation_time=2_000.0,
            seed=123,
        )

        assert isinstance(result, ParameterSweepResult)
        assert result.mu_rate == mu_rate
        assert result.capacity is None
        assert len(result.points) == len(lambda_rates)

        # In sorted order
        for i, l_rate in enumerate(lambda_rates):
            pt = result.points[i]
            assert pt.lambda_rate == l_rate
            assert pt.mu_rate == mu_rate
            assert pt.rho == l_rate / mu_rate
            assert pt.throughput > 0.0
            assert pt.mean_delay > 0.0
            assert pt.drop_probability == 0.0  # Infinite capacity has no drops
            assert 0.0 <= pt.server_utilization <= 1.0

        # Verify increasing load properties (rho, queue length, delay, utilization)
        for i in range(len(result.points) - 1):
            pt_low = result.points[i]
            pt_high = result.points[i + 1]

            # Throughput, queue length, delay, and utilization should grow with lambda
            assert pt_high.throughput > pt_low.throughput
            assert pt_high.mean_queue_length > pt_low.mean_queue_length
            assert pt_high.mean_delay > pt_low.mean_delay
            assert pt_high.mean_system_length > pt_low.mean_system_length
            assert pt_high.server_utilization > pt_low.server_utilization

    def test_parameter_sweep_finite_capacity_drops(self) -> None:
        """Verify finite M/M/1/K sweep tracks drops and handles lambda >= mu."""
        lambda_rates = [2.0, 5.0, 10.0, 20.0]
        mu_rate = 10.0
        capacity = 3
        result = run_parameter_sweep(
            lambda_rates=lambda_rates,
            mu_rate=mu_rate,
            capacity=capacity,
            simulation_time=2_000.0,
            seed=456,
        )

        assert result.capacity == capacity
        assert len(result.points) == len(lambda_rates)

        # Under heavy load (lambda=20, mu=10), drop probability should be significant
        pt_heavy = result.points[-1]
        assert pt_heavy.drop_probability > 0.20
        assert pt_heavy.throughput > 0.0
        assert 0.0 <= pt_heavy.server_utilization <= 1.0

        # Verify drop probability increases with arrival rate
        for i in range(len(result.points) - 1):
            assert result.points[i + 1].drop_probability >= result.points[i].drop_probability


class TestParameterSweepReport:
    """Verify report formatting contains expected structure and values."""

    def test_report_contains_table_headers_and_metrics(self) -> None:
        result = run_parameter_sweep(
            lambda_rates=[2.0, 4.0],
            mu_rate=10.0,
            capacity=None,
            simulation_time=200.0,
            seed=42,
        )
        report = format_parameter_sweep_report(result)

        assert "Parameter Sweep" in report
        assert "mu=10.000" in report
        assert "capacity=inf (M/M/1)" in report
        assert "Lambda" in report
        assert "Throughput" in report
        assert "Mean Delay" in report
        assert "Drop Prob" in report
        assert "Mean QLen" in report
        assert "Util" in report

        # Contains printed floats/metrics
        assert "2.000" in report
        assert "4.000" in report

    def test_report_contains_capacity_limit_when_finite(self) -> None:
        result = run_parameter_sweep(
            lambda_rates=[2.0, 4.0],
            mu_rate=10.0,
            capacity=5,
            simulation_time=200.0,
            seed=42,
        )
        report = format_parameter_sweep_report(result)
        assert "capacity=5" in report


class TestParameterSweepInputValidation:
    """Verify invalid parameters are rejected with clear error messages."""

    def test_rejects_negative_or_zero_mu(self) -> None:
        with pytest.raises(ValueError, match="mu_rate must be positive"):
            run_parameter_sweep(lambda_rates=[1.0], mu_rate=0.0)

        with pytest.raises(ValueError, match="mu_rate must be positive"):
            run_parameter_sweep(lambda_rates=[1.0], mu_rate=-5.0)

    def test_rejects_negative_or_zero_lambda(self) -> None:
        with pytest.raises(ValueError, match="arrival rates.*must be positive"):
            run_parameter_sweep(lambda_rates=[0.0], mu_rate=10.0)

        with pytest.raises(ValueError, match="arrival rates.*must be positive"):
            run_parameter_sweep(lambda_rates=[-1.0], mu_rate=10.0)

    def test_rejects_empty_lambdas(self) -> None:
        with pytest.raises(ValueError, match="lambda_rates must be a non-empty"):
            run_parameter_sweep(lambda_rates=[], mu_rate=10.0)

    def test_rejects_unstable_infinite_capacity_queue(self) -> None:
        # lambda >= mu is unstable for infinite capacity queue
        with pytest.raises(ValueError, match="Infinite capacity queue is unstable"):
            run_parameter_sweep(lambda_rates=[10.0], mu_rate=10.0, capacity=None)

        with pytest.raises(ValueError, match="Infinite capacity queue is unstable"):
            run_parameter_sweep(lambda_rates=[12.0], mu_rate=10.0, capacity=None)

    def test_rejects_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            run_parameter_sweep(lambda_rates=[1.0], mu_rate=10.0, capacity=0)


class TestParameterSweepVisualization:
    """Verify that plotting functions generate graphs correctly."""

    def test_plot_parameter_sweep_creates_images(self, tmpdir) -> None:
        """Verify plotting creates target images and saves them."""
        from validation.plots import plot_parameter_sweep

        result = run_parameter_sweep(
            lambda_rates=[2.0, 5.0],
            mu_rate=10.0,
            capacity=None,
            simulation_time=200.0,
            seed=42,
        )

        output_dir = str(tmpdir.mkdir("plots"))
        saved_paths = plot_parameter_sweep(result, output_dir=output_dir)

        assert len(saved_paths) == 3
        for path in saved_paths:
            import os
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            assert path.endswith(".png")

