"""Tests for M/M/1 theoretical validation."""

from __future__ import annotations

import pytest

from validation.mm1_validation import (
    format_mm1_validation_report,
    mm1_theoretical_metrics,
    run_mm1_validation,
)


def test_mm1_theoretical_metrics_match_closed_form() -> None:
    metrics = mm1_theoretical_metrics(lambda_rate=4.0, mu_rate=6.0)

    assert metrics.rho == pytest.approx(2.0 / 3.0)
    assert metrics.utilization == pytest.approx(2.0 / 3.0)
    assert metrics.L == pytest.approx(2.0)
    assert metrics.Lq == pytest.approx(4.0 / 3.0)
    assert metrics.W == pytest.approx(0.5)
    assert metrics.Wq == pytest.approx(1.0 / 3.0)


def test_mm1_theoretical_metrics_reject_unstable_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="lambda_rate must be positive"):
        mm1_theoretical_metrics(lambda_rate=0.0, mu_rate=1.0)
    with pytest.raises(ValueError, match="mu_rate must be positive"):
        mm1_theoretical_metrics(lambda_rate=1.0, mu_rate=0.0)
    with pytest.raises(ValueError, match="lambda_rate < mu_rate"):
        mm1_theoretical_metrics(lambda_rate=2.0, mu_rate=2.0)


def test_run_mm1_validation_produces_close_observed_values() -> None:
    result = run_mm1_validation(
        lambda_rate=4.0,
        mu_rate=6.0,
        simulation_time=20_000.0,
        seed=42,
        relative_error_threshold=0.10,
    )

    assert result.observed.completed_packets > 1_000
    assert result.observed.throughput > 0.0
    assert len(result.comparisons) == 5
    assert result.passed is True


def test_report_formatter_contains_key_fields() -> None:
    result = run_mm1_validation(
        lambda_rate=4.0,
        mu_rate=6.0,
        simulation_time=5_000.0,
        seed=123,
        relative_error_threshold=0.10,
    )
    report = format_mm1_validation_report(result)

    assert "M/M/1 Validation" in report
    assert "lambda=4.000" in report
    assert "mu=6.000" in report
    assert "utilization" in report
    assert "Observed support:" in report
