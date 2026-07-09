#!/usr/bin/env python3
"""Execute parameter sweeps and generate visualization plots.

Runs two comparative sweeps:
  1. M/M/1 infinite queue (lambda 1..9, mu = 10)
  2. M/M/1/5 finite queue (lambda 1..20, mu = 10)

Plots the performance metrics of both systems side-by-side or overlaid.
"""

from __future__ import annotations

import os
import sys

# Ensure project source directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from validation.parameter_sweep import run_parameter_sweep, format_parameter_sweep_report
from validation.plots import plot_parameter_sweep


def main() -> None:
    print("Running parameter sweeps...")

    # 1. Sweep M/M/1 (Infinite capacity queue, stable up to lambda = 9)
    print("Executing M/M/1 (infinite capacity) sweep (lambda = 1..9)...")
    mm1_result = run_parameter_sweep(
        lambda_rates=range(1, 10),
        mu_rate=10.0,
        capacity=None,
        simulation_time=10_000.0,
        seed=101,
    )

    # 2. Sweep M/M/1/5 (Finite capacity queue, capacity = 5, stable at all rates due to drops)
    print("Executing M/M/1/5 (finite capacity = 5) sweep (lambda = 1..20)...")
    mm1k_result = run_parameter_sweep(
        lambda_rates=range(1, 21),
        mu_rate=10.0,
        capacity=5,
        simulation_time=10_000.0,
        seed=202,
    )

    # Print reports to stdout
    print("\n" + "=" * 95)
    print(format_parameter_sweep_report(mm1_result))
    print("=" * 95)
    print(format_parameter_sweep_report(mm1k_result))
    print("=" * 95 + "\n")

    # Generate comparative plots
    output_dir = "plots"
    print(f"Generating and saving plots in '{output_dir}/' directory...")
    saved_paths = plot_parameter_sweep(
        sweeps={
            "M/M/1 (infinite capacity)": mm1_result,
            "M/M/1/5 (capacity = 5)": mm1k_result,
        },
        output_dir=output_dir,
    )

    print("Successfully generated the following plots:")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
