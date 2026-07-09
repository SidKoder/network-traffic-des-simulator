"""Parameter sweeps for simulator performance exploration under varying load.

Sweeps the arrival rate (lambda) of a queueing system while keeping the
service rate (mu) constant, recording metrics to evaluate behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from analytics.system_statistics import compute_system_statistics
from validation.mm1_validation import (
    _mm1_observed_metrics,
    _simulate_mm1_records,
)


@dataclass(frozen=True)
class SweepPoint:
    """Performance metrics at a specific workload intensity."""

    lambda_rate: float
    mu_rate: float
    rho: float
    throughput: float
    mean_delay: float  # W (mean system time)
    mean_wait: float  # Wq (mean wait time in queue)
    drop_probability: float
    mean_queue_length: float  # Lq (mean queue size)
    mean_system_length: float  # L (mean packets in system)
    server_utilization: float


@dataclass(frozen=True)
class ParameterSweepResult:
    """Collection of sweep points representing system performance under load."""

    mu_rate: float
    capacity: int | None
    points: tuple[SweepPoint, ...]


def run_parameter_sweep(
    *,
    lambda_rates: Iterable[float],
    mu_rate: float,
    capacity: int | None = None,
    simulation_time: float = 10_000.0,
    seed: int = 42,
) -> ParameterSweepResult:
    """Execute a parameter sweep over arrival rates (lambda) for a given mu.

    Runs a discrete event simulation for each lambda rate in `lambda_rates`,
    keeping `mu_rate` constant.

    Parameters:
        lambda_rates: Arrival rates to sweep (must be positive).
        mu_rate: Constant service rate (must be positive).
        capacity: Queue capacity limit (None for infinite / M/M/1).
        simulation_time: Duration of each simulation run.
        seed: Random seed for reproducibility.

    Returns:
        A `ParameterSweepResult` containing a `SweepPoint` for each run.

    Raises:
        ValueError: For invalid parameter ranges or unstable configurations.
    """
    if mu_rate <= 0.0:
        raise ValueError("mu_rate must be positive")
    if simulation_time <= 0.0:
        raise ValueError("simulation_time must be positive")

    lambdas = sorted(list(lambda_rates))
    if not lambdas:
        raise ValueError("lambda_rates must be a non-empty sequence")

    for l_rate in lambdas:
        if l_rate <= 0.0:
            raise ValueError("arrival rates (lambda) must be positive")
        if capacity is None and l_rate >= mu_rate:
            raise ValueError(
                f"Infinite capacity queue is unstable when lambda ({l_rate}) >= mu ({mu_rate})"
            )

    if capacity is not None and capacity < 1:
        raise ValueError("capacity must be >= 1 when specified")

    points: list[SweepPoint] = []
    rng = np.random.default_rng(seed)

    for l_rate in lambdas:
        # Generate a distinct seed per sweep run to ensure runs are independent
        # but deterministic for the sweep sequence.
        run_seed = int(rng.integers(1, 1_000_000_000))
        records = _simulate_mm1_records(
            lambda_rate=l_rate,
            mu_rate=mu_rate,
            simulation_time=simulation_time,
            seed=run_seed,
            capacity=capacity,
        )
        stats = compute_system_statistics(records)
        observed = _mm1_observed_metrics(records)

        points.append(
            SweepPoint(
                lambda_rate=l_rate,
                mu_rate=mu_rate,
                rho=l_rate / mu_rate,
                throughput=observed.throughput,
                mean_delay=observed.W,
                mean_wait=observed.Wq,
                drop_probability=stats.drop_probability,
                mean_queue_length=observed.Lq,
                mean_system_length=observed.L,
                server_utilization=observed.utilization,
            )
        )

    return ParameterSweepResult(
        mu_rate=mu_rate,
        capacity=capacity,
        points=tuple(points),
    )


def format_parameter_sweep_report(result: ParameterSweepResult) -> str:
    """Format a CLI-friendly table presenting the parameter sweep results.

    Displays inputs (lambda, rho) alongside performance KPIs for each run.
    """
    cap_str = f"capacity={result.capacity}" if result.capacity is not None else "capacity=inf (M/M/1)"
    lines = [
        f"Parameter Sweep (mu={result.mu_rate:.3f}, {cap_str})",
        "",
        f"{'Lambda':>8}  {'Rho':>6}  {'Throughput':>10}  {'Mean Delay':>10}  {'Mean Wait':>9}  {'Drop Prob':>9}  {'Mean QLen':>9}  {'Mean SysLen':>11}  {'Util':>6}",
        "-" * 95,
    ]

    for pt in result.points:
        lines.append(
            f"{pt.lambda_rate:>8.3f}  "
            f"{pt.rho:>6.3f}  "
            f"{pt.throughput:>10.4f}  "
            f"{pt.mean_delay:>10.4f}  "
            f"{pt.mean_wait:>9.4f}  "
            f"{pt.drop_probability:>9.4%}  "
            f"{pt.mean_queue_length:>9.4f}  "
            f"{pt.mean_system_length:>11.4f}  "
            f"{pt.server_utilization:>6.2%}"
        )

    return "\n".join(lines)
