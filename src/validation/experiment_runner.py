"""Research experiment runner framework for coordinating queueing simulations.

Orchestrates running simulations, collecting performance metrics, storing results
as machine-readable JSON, generating comparative plots, and rendering research reports.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from validation.convergence_experiment import run_convergence_experiment, format_convergence_report
from validation.mm1_validation import run_mm1_validation, format_mm1_validation_report
from validation.parameter_sweep import run_parameter_sweep, format_parameter_sweep_report
from validation.plots import plot_parameter_sweep


class ExperimentType(str, Enum):
    """Supported research experiment types."""

    SINGLE_RUN = "single_run"
    CONVERGENCE = "convergence"
    PARAMETER_SWEEP = "parameter_sweep"


class ExperimentConfig(BaseModel):
    """Unified schema governing a research simulation experiment."""

    name: str = Field(description="Unique experiment name")
    type: ExperimentType = Field(description="Type of experiment to execute")
    mu_rate: float = Field(gt=0.0, description="Service rate (mu)")
    lambda_rate: float | None = Field(default=None, gt=0.0, description="Point arrival rate (for single_run/convergence)")
    lambda_rates: list[float] | None = Field(default=None, description="Arrival rates to sweep (for parameter_sweep)")
    capacity: int | None = Field(default=None, ge=1, description="Queue capacity; None for infinite")
    simulation_time: float | None = Field(default=None, gt=0.0, description="Simulation duration (for single_run/parameter_sweep)")
    durations: list[float] | None = Field(default=None, description="Simulation durations (for convergence)")
    num_replications: int = Field(default=1, ge=1, description="Number of replications per point (for convergence)")
    metric_names: list[str] | None = Field(default=None, description="Metrics to track (for convergence)")
    seed: int = Field(default=42, description="Base seed for reproducibility")
    output_dir: str | None = Field(default=None, description="Output directory; defaults to results/<name>")

    @model_validator(mode="after")
    def validate_experiment_parameters(self) -> ExperimentConfig:
        """Validate fields depending on the chosen experiment type."""
        if self.type == ExperimentType.SINGLE_RUN:
            if self.lambda_rate is None:
                raise ValueError("lambda_rate is required for single_run experiments")
            if self.simulation_time is None:
                raise ValueError("simulation_time is required for single_run experiments")
            if self.capacity is None and self.lambda_rate >= self.mu_rate:
                raise ValueError("Unstable M/M/1 queue parameters (lambda_rate >= mu_rate)")

        elif self.type == ExperimentType.CONVERGENCE:
            if self.lambda_rate is None:
                raise ValueError("lambda_rate is required for convergence experiments")
            if self.capacity is None and self.lambda_rate >= self.mu_rate:
                raise ValueError("Unstable M/M/1 queue parameters (lambda_rate >= mu_rate)")

        elif self.type == ExperimentType.PARAMETER_SWEEP:
            if not self.lambda_rates:
                raise ValueError("lambda_rates must be provided and non-empty for parameter sweeps")
            if self.simulation_time is None:
                raise ValueError("simulation_time is required for parameter sweep experiments")
            for rate in self.lambda_rates:
                if rate <= 0.0:
                    raise ValueError("All sweep arrival rates must be positive")
                if self.capacity is None and rate >= self.mu_rate:
                    raise ValueError(
                        f"Unstable sweep rate: arrival rate ({rate}) >= mu_rate ({self.mu_rate}) "
                        "for infinite capacity queue"
                    )

        return self


class ExperimentRunner:
    """Orchestrates simulation execution, metrics storage, and reporting."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize with an experiment configuration schema."""
        self.config = config
        # Resolve output directory
        if self.config.output_dir:
            self.output_dir = Path(self.config.output_dir)
        else:
            self.output_dir = Path("results") / self.config.name

    def execute(self) -> dict[str, Any]:
        """Execute the configured simulation experiment and write outputs to disk.

        Returns:
            Dictionary containing raw results that were saved.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "plots", exist_ok=True)

        # Save config copy for reproducibility records
        self._save_config_provenance()

        # Run appropriate experiment type
        if self.config.type == ExperimentType.SINGLE_RUN:
            data = self._execute_single_run()
        elif self.config.type == ExperimentType.CONVERGENCE:
            data = self._execute_convergence()
        else:
            data = self._execute_parameter_sweep()

        # Save structured results to JSON
        with open(self.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Generate report.md
        self._generate_markdown_report(data)

        return data

    def _save_config_provenance(self) -> None:
        """Save a YAML copy of the configuration in the results directory."""
        config_dict = self.config.model_dump()
        # Serialize enum to string
        config_dict["type"] = config_dict["type"].value
        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    def _execute_single_run(self) -> dict[str, Any]:
        """Execute a single-run configuration validation."""
        assert self.config.lambda_rate is not None
        assert self.config.simulation_time is not None

        # Reuse run_mm1_validation under the hood
        validation_result = run_mm1_validation(
            lambda_rate=self.config.lambda_rate,
            mu_rate=self.config.mu_rate,
            simulation_time=self.config.simulation_time,
            seed=self.config.seed,
        )

        # Map to structured dictionary
        result_data = {
            "name": self.config.name,
            "type": self.config.type.value,
            "parameters": {
                "lambda_rate": self.config.lambda_rate,
                "mu_rate": self.config.mu_rate,
                "capacity": self.config.capacity,
                "simulation_time": self.config.simulation_time,
                "seed": self.config.seed,
            },
            "metrics": {
                "observed": {
                    "completed_packets": validation_result.observed.completed_packets,
                    "throughput": validation_result.observed.throughput,
                    "utilization": validation_result.observed.utilization,
                    "L": validation_result.observed.L,
                    "Lq": validation_result.observed.Lq,
                    "W": validation_result.observed.W,
                    "Wq": validation_result.observed.Wq,
                },
                "theoretical": {
                    "rho": validation_result.theoretical.rho,
                    "utilization": validation_result.theoretical.utilization,
                    "L": validation_result.theoretical.L,
                    "Lq": validation_result.theoretical.Lq,
                    "W": validation_result.theoretical.W,
                    "Wq": validation_result.theoretical.Wq,
                },
                "comparisons": [
                    {
                        "name": cmp.name,
                        "theoretical": cmp.theoretical,
                        "observed": cmp.observed,
                        "relative_error": cmp.relative_error,
                        "within_threshold": cmp.within_threshold,
                    }
                    for cmp in validation_result.comparisons
                ],
                "passed": validation_result.passed,
            },
        }
        return result_data

    def _execute_convergence(self) -> dict[str, Any]:
        """Execute convergence sweeps across durations."""
        assert self.config.lambda_rate is not None

        kwargs: dict[str, Any] = {}
        if self.config.durations:
            kwargs["durations"] = self.config.durations
        if self.config.num_replications:
            kwargs["num_replications"] = self.config.num_replications
        if self.config.metric_names:
            kwargs["metric_names"] = self.config.metric_names

        conv_result = run_convergence_experiment(
            lambda_rate=self.config.lambda_rate,
            mu_rate=self.config.mu_rate,
            seed=self.config.seed,
            **kwargs,
        )

        series_data = []
        for series in conv_result.metric_series:
            points_list = [
                {
                    "duration": pt.duration,
                    "observed": pt.observed,
                    "theoretical": pt.theoretical,
                    "relative_error": pt.relative_error,
                }
                for pt in series.points
            ]
            series_data.append(
                {
                    "metric_name": series.metric_name,
                    "converged": series.converged,
                    "log_log_slope": series.log_log_slope,
                    "points": points_list,
                }
            )

        result_data = {
            "name": self.config.name,
            "type": self.config.type.value,
            "parameters": {
                "lambda_rate": self.config.lambda_rate,
                "mu_rate": self.config.mu_rate,
                "capacity": self.config.capacity,
                "num_replications": self.config.num_replications,
                "seed": self.config.seed,
            },
            "converged": conv_result.converged,
            "series": series_data,
            "report_text": format_convergence_report(conv_result),
        }
        return result_data

    def _execute_parameter_sweep(self) -> dict[str, Any]:
        """Execute arrival rate parameter sweeps."""
        assert self.config.lambda_rates is not None
        assert self.config.simulation_time is not None

        sweep_result = run_parameter_sweep(
            lambda_rates=self.config.lambda_rates,
            mu_rate=self.config.mu_rate,
            capacity=self.config.capacity,
            simulation_time=self.config.simulation_time,
            seed=self.config.seed,
        )

        # Generate comparative charts
        plot_parameter_sweep(sweep_result, output_dir=str(self.output_dir / "plots"))

        points_list = []
        for pt in sweep_result.points:
            points_list.append(
                {
                    "lambda_rate": pt.lambda_rate,
                    "mu_rate": pt.mu_rate,
                    "rho": pt.rho,
                    "throughput": pt.throughput,
                    "mean_delay": pt.mean_delay,
                    "mean_wait": pt.mean_wait,
                    "drop_probability": pt.drop_probability,
                    "mean_queue_length": pt.mean_queue_length,
                    "mean_system_length": pt.mean_system_length,
                    "server_utilization": pt.server_utilization,
                }
            )

        result_data = {
            "name": self.config.name,
            "type": self.config.type.value,
            "parameters": {
                "mu_rate": self.config.mu_rate,
                "capacity": self.config.capacity,
                "simulation_time": self.config.simulation_time,
                "seed": self.config.seed,
            },
            "points": points_list,
            "report_text": format_parameter_sweep_report(sweep_result),
        }
        return result_data

    def _generate_markdown_report(self, data: dict[str, Any]) -> None:
        """Render a research Markdown report detailing experiment findings."""
        lines = [
            f"# Simulation Research Report: {self.config.name}",
            "",
            "## Experiment Meta",
            f"- **Type**: `{self.config.type.value}`",
            f"- **Constant Service Rate (mu)**: `{self.config.mu_rate}`",
            f"- **Queue Capacity Limit**: `{self.config.capacity if self.config.capacity is not None else 'Infinite (M/M/1)'}`",
            f"- **Base Random Seed**: `{self.config.seed}`",
        ]

        if self.config.type == ExperimentType.SINGLE_RUN:
            p = data["parameters"]
            m = data["metrics"]
            lines.extend(
                [
                    f"- **Arrival Rate (lambda)**: `{p['lambda_rate']}`",
                    f"- **Simulated Duration**: `{p['simulation_time']}`",
                    "",
                    "## Validation Outcomes",
                    f"Overall Pass Status: **{'PASS' if m['passed'] else 'FAIL'}**",
                    "",
                    "| Metric | Theoretical | Observed | Rel. Error | Passes Threshold |",
                    "|---|---|---|---|---|",
                ]
            )
            for cmp in m["comparisons"]:
                verdict = "✅ Yes" if cmp["within_threshold"] else "❌ No"
                lines.append(
                    f"| {cmp['name']} | {cmp['theoretical']:.4f} | {cmp['observed']:.4f} | {cmp['relative_error']:.2%} | {verdict} |"
                )

            lines.extend(
                [
                    "",
                    f"Observed completed packets: `{m['observed']['completed_packets']}`",
                    f"Observed throughput: `{m['observed']['throughput']:.4f}`",
                ]
            )

        elif self.config.type == ExperimentType.CONVERGENCE:
            p = data["parameters"]
            lines.extend(
                [
                    f"- **Arrival Rate (lambda)**: `{p['lambda_rate']}`",
                    f"- **Replications per duration**: `{p['num_replications']}`",
                    f"- **Overall Convergence Passed**: **{'YES' if data['converged'] else 'NO'}**",
                    "",
                    "## Convergence Summaries",
                    "| Metric | Log-Log Trend Slope | Convergence Status |",
                    "|---|---|---|",
                ]
            )
            for series in data["series"]:
                verdict = "✅ Converged" if series["converged"] else "❌ Oscillating/Stale"
                lines.append(
                    f"| {series['metric_name']} | {series['log_log_slope']:+.4f} | {verdict} |"
                )

            lines.extend(
                [
                    "",
                    "### Details by Step",
                    "```",
                    data["report_text"],
                    "```",
                ]
            )

        elif self.config.type == ExperimentType.PARAMETER_SWEEP:
            p = data["parameters"]
            lines.extend(
                [
                    f"- **Simulated Duration**: `{p['simulation_time']}`",
                    "",
                    "## Performance Sweep Details",
                    "```",
                    data["report_text"],
                    "```",
                    "",
                    "## Visualization Charts",
                    "Comparative parameter sweep visualization charts:",
                    "",
                    "### 1. Workload vs. Average Delay",
                    "![Arrival Rate vs Delay](plots/arrival_vs_delay.png)",
                    "",
                    "### 2. Workload vs. Drop Probability",
                    "![Arrival Rate vs Drop Probability](plots/arrival_vs_drop_prob.png)",
                    "",
                    "### 3. Workload vs. Server Utilization",
                    "![Arrival Rate vs Server Utilization](plots/arrival_vs_utilization.png)",
                ]
            )

        # Write to report.md
        with open(self.output_dir / "report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
