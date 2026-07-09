"""Monte Carlo convergence experiments for M/M/1 simulator validation.

Runs the simulator at geometrically increasing durations and verifies that
the gap between observed and theoretical steady-state metrics shrinks —
the hallmark of a correct, unbiased discrete-event simulator.

The convergence criterion is noise-tolerant: rather than requiring strict
monotonic decrease (which random fluctuations can violate), it checks that
the final error is strictly less than the first AND that a log-log linear
regression of error vs. duration has negative slope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from validation.mm1_validation import (
    _mm1_observed_metrics,
    _simulate_mm1_records,
    mm1_theoretical_metrics,
)
from validation.theoretical_metrics import mean_relative_error

DEFAULT_DURATIONS: tuple[float, ...] = (100.0, 500.0, 2_000.0, 10_000.0, 50_000.0)
DEFAULT_METRIC_NAMES: tuple[str, ...] = ("W", "Wq", "utilization", "L", "Lq")


@dataclass(frozen=True)
class ConvergencePoint:
    """One measurement at a single simulation duration."""

    duration: float
    seed: int
    observed: float
    theoretical: float
    relative_error: float


@dataclass(frozen=True)
class ConvergenceMetricSeries:
    """Convergence trajectory for a single metric across all durations.

    Attributes:
        metric_name: Which M/M/1 metric this series tracks.
        points: One ``ConvergencePoint`` per duration, in increasing order.
        converged: Whether the noise-tolerant convergence criterion is met.
        log_log_slope: Slope of ``log(error) vs log(duration)`` regression.
            Negative slope indicates convergence.
    """

    metric_name: str
    points: tuple[ConvergencePoint, ...]
    converged: bool
    log_log_slope: float


@dataclass(frozen=True)
class ConvergenceResult:
    """Full result bundle for one convergence experiment.

    Attributes:
        lambda_rate: Poisson arrival rate used.
        mu_rate: Exponential service rate used.
        num_replications: Number of independent replications per duration.
        metric_series: One ``ConvergenceMetricSeries`` per tracked metric.
        converged: Whether ALL metrics passed the convergence criterion.
    """

    lambda_rate: float
    mu_rate: float
    num_replications: int
    metric_series: tuple[ConvergenceMetricSeries, ...]

    @property
    def converged(self) -> bool:
        """Return whether every tracked metric has converged."""
        return all(series.converged for series in self.metric_series)


def run_convergence_experiment(
    *,
    lambda_rate: float,
    mu_rate: float,
    durations: tuple[float, ...] | list[float] = DEFAULT_DURATIONS,
    seed: int = 42,
    num_replications: int = 1,
    metric_names: tuple[str, ...] | list[str] = DEFAULT_METRIC_NAMES,
) -> ConvergenceResult:
    """Run an M/M/1 convergence experiment across increasing durations.

    For each duration, the simulator is run ``num_replications`` times with
    seeds ``seed, seed+1, …, seed+num_replications-1``.  The relative error
    for each metric is the **mean** across replications.

    Convergence is assessed with a noise-tolerant criterion per metric:
      1. The final relative error must be strictly less than the first.
      2. A least-squares fit of ``log(error) vs log(duration)`` must have
         a negative slope — confirming an overall downward trend even if
         individual steps occasionally increase due to randomness.

    Parameters:
        lambda_rate: Poisson arrival rate (must be positive and < mu_rate).
        mu_rate: Exponential service rate (must be positive).
        durations: Simulation durations to sweep, in ascending order.
        seed: Base random seed (each replication offsets by its index).
        num_replications: Independent runs per duration (≥ 1).
        metric_names: Which M/M/1 metrics to track.

    Returns:
        A ``ConvergenceResult`` with per-metric convergence trajectories.

    Raises:
        ValueError: On invalid rates, empty durations, or bad replications.
    """
    if lambda_rate <= 0.0:
        raise ValueError("lambda_rate must be positive")
    if mu_rate <= 0.0:
        raise ValueError("mu_rate must be positive")
    if lambda_rate >= mu_rate:
        raise ValueError("lambda_rate must be less than mu_rate for stable M/M/1")
    if not durations:
        raise ValueError("durations must be a non-empty sequence")
    if num_replications < 1:
        raise ValueError("num_replications must be >= 1")
    if not metric_names:
        raise ValueError("metric_names must be a non-empty sequence")

    durations = tuple(sorted(durations))
    theoretical = mm1_theoretical_metrics(lambda_rate, mu_rate)

    # Map metric names to their theoretical values.
    theoretical_values = {
        "W": theoretical.W,
        "Wq": theoretical.Wq,
        "utilization": theoretical.utilization,
        "L": theoretical.L,
        "Lq": theoretical.Lq,
    }
    for name in metric_names:
        if name not in theoretical_values:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Valid names: {sorted(theoretical_values.keys())}"
            )

    # Collect per-metric, per-duration relative errors.
    # Shape: {metric_name: [(duration, mean_rel_error, mean_observed), ...]}
    metric_trajectories: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in metric_names
    }

    for duration in durations:
        # Accumulate errors across replications.
        replication_errors: dict[str, list[float]] = {
            name: [] for name in metric_names
        }
        replication_observed: dict[str, list[float]] = {
            name: [] for name in metric_names
        }

        for rep_idx in range(num_replications):
            rep_seed = seed + rep_idx
            records = _simulate_mm1_records(
                lambda_rate=lambda_rate,
                mu_rate=mu_rate,
                simulation_time=duration,
                seed=rep_seed,
            )
            observed = _mm1_observed_metrics(records)

            observed_values = {
                "W": observed.W,
                "Wq": observed.Wq,
                "utilization": observed.utilization,
                "L": observed.L,
                "Lq": observed.Lq,
            }

            for name in metric_names:
                rel_err = mean_relative_error(
                    theoretical_values[name], observed_values[name]
                )
                replication_errors[name].append(rel_err)
                replication_observed[name].append(observed_values[name])

        for name in metric_names:
            mean_err = float(np.mean(replication_errors[name]))
            mean_obs = float(np.mean(replication_observed[name]))
            metric_trajectories[name].append((duration, mean_err, mean_obs))

    # Build ConvergenceMetricSeries per metric.
    all_series: list[ConvergenceMetricSeries] = []

    for name in metric_names:
        trajectory = metric_trajectories[name]
        points = tuple(
            ConvergencePoint(
                duration=dur,
                seed=seed,
                observed=obs,
                theoretical=theoretical_values[name],
                relative_error=err,
            )
            for dur, err, obs in trajectory
        )

        errors = [p.relative_error for p in points]
        log_log_slope = _log_log_slope(
            [p.duration for p in points], errors
        )
        converged = _check_convergence(errors, log_log_slope)

        all_series.append(
            ConvergenceMetricSeries(
                metric_name=name,
                points=points,
                converged=converged,
                log_log_slope=log_log_slope,
            )
        )

    return ConvergenceResult(
        lambda_rate=lambda_rate,
        mu_rate=mu_rate,
        num_replications=num_replications,
        metric_series=tuple(all_series),
    )


def _log_log_slope(durations: list[float], errors: list[float]) -> float:
    """Compute the slope of log(error) vs log(duration) via least-squares.

    Returns ``0.0`` if any error is zero or negative (degenerate case).
    """
    if len(durations) < 2:
        return 0.0

    # Filter out zero/negative errors that can't be log-transformed.
    valid = [(d, e) for d, e in zip(durations, errors) if e > 0.0]
    if len(valid) < 2:
        return 0.0

    log_d = np.array([np.log(d) for d, _ in valid])
    log_e = np.array([np.log(e) for _, e in valid])

    # Least-squares slope: Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
    d_mean = np.mean(log_d)
    e_mean = np.mean(log_e)
    numerator = float(np.sum((log_d - d_mean) * (log_e - e_mean)))
    denominator = float(np.sum((log_d - d_mean) ** 2))

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def _check_convergence(errors: list[float], log_log_slope: float) -> bool:
    """Noise-tolerant convergence criterion.

    Requires BOTH:
      1. The final relative error is strictly less than the first.
      2. The log-log regression slope is negative (overall downward trend).
    """
    if len(errors) < 2:
        return False

    final_less_than_first = errors[-1] < errors[0]
    slope_is_negative = log_log_slope < 0.0

    return final_less_than_first and slope_is_negative


def format_convergence_report(result: ConvergenceResult) -> str:
    """Format a CLI-friendly convergence experiment report.

    Produces a table with one row per duration and one column per metric,
    showing relative error at each duration step.  Includes a summary
    footer with per-metric convergence verdict and log-log slope.
    """
    status = "PASS" if result.converged else "FAIL"
    lines = [
        (
            f"Convergence Experiment"
            f" (lambda={result.lambda_rate:.3f},"
            f" mu={result.mu_rate:.3f},"
            f" replications={result.num_replications})"
        ),
        f"Status: {status}",
        "",
    ]

    # Header row.
    metric_names = [s.metric_name for s in result.metric_series]
    header = f"{'Duration':>12}"
    for name in metric_names:
        header += f"  {name:>14}"
    lines.append(header)
    lines.append("-" * len(header))

    # One row per duration.
    num_durations = len(result.metric_series[0].points)
    for i in range(num_durations):
        duration = result.metric_series[0].points[i].duration
        row = f"{duration:>12,.0f}"
        for series in result.metric_series:
            err = series.points[i].relative_error
            row += f"  {err:>13.2%}"
        lines.append(row)

    # Summary footer.
    lines.append("")
    lines.append("Per-metric summary:")
    for series in result.metric_series:
        verdict = "CONVERGED" if series.converged else "NOT CONVERGED"
        lines.append(
            f"  {series.metric_name:<14} slope={series.log_log_slope:+.4f}  {verdict}"
        )

    return "\n".join(lines)
