"""Tests for the research experiment runner framework."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from validation.experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentType,
)

# Mark all tests in this module with the 'runner' marker
pytestmark = pytest.mark.runner


class TestExperimentConfigValidation:
    """Verify that experiment configs validate correctly."""

    def test_valid_single_run_config(self) -> None:
        cfg = ExperimentConfig(
            name="test_single",
            type=ExperimentType.SINGLE_RUN,
            mu_rate=10.0,
            lambda_rate=5.0,
            simulation_time=100.0,
        )
        assert cfg.name == "test_single"
        assert cfg.type == ExperimentType.SINGLE_RUN

    def test_single_run_missing_lambda(self) -> None:
        with pytest.raises(ValidationError, match="lambda_rate is required"):
            ExperimentConfig(
                name="test_single",
                type=ExperimentType.SINGLE_RUN,
                mu_rate=10.0,
                simulation_time=100.0,
            )

    def test_single_run_missing_simulation_time(self) -> None:
        with pytest.raises(ValidationError, match="simulation_time is required"):
            ExperimentConfig(
                name="test_single",
                type=ExperimentType.SINGLE_RUN,
                mu_rate=10.0,
                lambda_rate=5.0,
            )

    def test_single_run_unstable_mm1(self) -> None:
        with pytest.raises(ValidationError, match="Unstable M/M/1 queue"):
            ExperimentConfig(
                name="test_single",
                type=ExperimentType.SINGLE_RUN,
                mu_rate=10.0,
                lambda_rate=10.0,
            )

    def test_valid_convergence_config(self) -> None:
        cfg = ExperimentConfig(
            name="test_conv",
            type=ExperimentType.CONVERGENCE,
            mu_rate=10.0,
            lambda_rate=5.0,
            durations=[100, 200],
            num_replications=2,
            metric_names=["W"],
        )
        assert cfg.type == ExperimentType.CONVERGENCE

    def test_valid_parameter_sweep_config(self) -> None:
        cfg = ExperimentConfig(
            name="test_sweep",
            type=ExperimentType.PARAMETER_SWEEP,
            mu_rate=10.0,
            lambda_rates=[2.0, 4.0, 6.0],
            simulation_time=500.0,
        )
        assert cfg.type == ExperimentType.PARAMETER_SWEEP

    def test_parameter_sweep_missing_lambda_rates(self) -> None:
        with pytest.raises(ValidationError, match="lambda_rates must be provided"):
            ExperimentConfig(
                name="test_sweep",
                type=ExperimentType.PARAMETER_SWEEP,
                mu_rate=10.0,
                simulation_time=500.0,
            )


class TestExperimentRunnerExecution:
    """Verify that the runner executes simulations and saves structured files."""

    def test_single_run_creates_outputs(self, tmpdir) -> None:
        cfg = ExperimentConfig(
            name="test_single_run",
            type=ExperimentType.SINGLE_RUN,
            mu_rate=10.0,
            lambda_rate=4.0,
            simulation_time=200.0,
            seed=42,
            output_dir=str(tmpdir / "single_run"),
        )
        runner = ExperimentRunner(cfg)
        results = runner.execute()

        assert results["name"] == "test_single_run"
        assert results["metrics"]["passed"] is True

        # Check files
        out_dir = Path(runner.output_dir)
        assert (out_dir / "config.yaml").exists()
        assert (out_dir / "results.json").exists()
        assert (out_dir / "report.md").exists()

        with open(out_dir / "results.json", "r") as f:
            data = json.load(f)
            assert data["type"] == "single_run"
            assert "metrics" in data

    def test_convergence_run_creates_outputs(self, tmpdir) -> None:
        cfg = ExperimentConfig(
            name="test_conv_run",
            type=ExperimentType.CONVERGENCE,
            mu_rate=10.0,
            lambda_rate=4.0,
            durations=[50.0, 100.0],
            num_replications=1,
            metric_names=["W"],
            seed=42,
            output_dir=str(tmpdir / "convergence"),
        )
        runner = ExperimentRunner(cfg)
        results = runner.execute()

        assert results["name"] == "test_conv_run"

        out_dir = Path(runner.output_dir)
        assert (out_dir / "config.yaml").exists()
        assert (out_dir / "results.json").exists()
        assert (out_dir / "report.md").exists()

        with open(out_dir / "results.json", "r") as f:
            data = json.load(f)
            assert data["type"] == "convergence"
            assert "series" in data

    def test_parameter_sweep_creates_outputs(self, tmpdir) -> None:
        cfg = ExperimentConfig(
            name="test_sweep_run",
            type=ExperimentType.PARAMETER_SWEEP,
            mu_rate=10.0,
            lambda_rates=[2.0, 5.0],
            simulation_time=200.0,
            seed=42,
            output_dir=str(tmpdir / "sweep"),
        )
        runner = ExperimentRunner(cfg)
        results = runner.execute()

        assert results["name"] == "test_sweep_run"

        out_dir = Path(runner.output_dir)
        assert (out_dir / "config.yaml").exists()
        assert (out_dir / "results.json").exists()
        assert (out_dir / "report.md").exists()

        # Check visual plots are generated
        plots_dir = out_dir / "plots"
        assert plots_dir.exists()
        assert (plots_dir / "arrival_vs_delay.png").exists()
        assert (plots_dir / "arrival_vs_drop_prob.png").exists()
        assert (plots_dir / "arrival_vs_utilization.png").exists()


# Import helper utilities for Path and json
import json
from pathlib import Path
