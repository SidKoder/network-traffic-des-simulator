"""Matplotlib-based plotting layer for simulator parameter sweeps.

Generates themed, publication-quality performance graphs to visualize performance
under varying workloads, comparing infinite vs finite queueing systems.
"""

from __future__ import annotations

import os

# Set matplotlib backend to Agg to run headlessly in all environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from validation.parameter_sweep import ParameterSweepResult


def plot_parameter_sweep(
    sweeps: ParameterSweepResult | dict[str, ParameterSweepResult],
    output_dir: str = "plots",
) -> list[str]:
    """Generate and save comparative graphs for parameter sweep results.

    Creates three graphs in `output_dir`:
      1. Arrival Rate vs. Average Delay (W)
      2. Arrival Rate vs. Drop Probability
      3. Arrival Rate vs. Server Utilization

    Parameters:
        sweeps: A single ParameterSweepResult or a dict mapping labels to
            ParameterSweepResult objects (e.g. {"M/M/1 (inf)": res1, "M/M/1/5": res2}).
        output_dir: Directory where the plotted image files will be saved.

    Returns:
        List of absolute file paths to the generated plots.
    """
    if isinstance(sweeps, ParameterSweepResult):
        label = "M/M/1" if sweeps.capacity is None else f"M/M/1/{sweeps.capacity}"
        sweep_dict = {label: sweeps}
    else:
        sweep_dict = sweeps

    os.makedirs(output_dir, exist_ok=True)
    generated_paths: list[str] = []

    # Curated premium color palette (e.g., Google HSL tailors/harmonious)
    colors = ["#1a73e8", "#ea4335", "#f9ab00", "#12b5cb", "#9334e6"]

    # 1. Arrival Rate vs Average Delay
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, sweep), color in zip(sweep_dict.items(), colors):
        lambdas = [pt.lambda_rate for pt in sweep.points]
        delays = [pt.mean_delay for pt in sweep.points]
        ax.plot(lambdas, delays, marker="o", linewidth=2, color=color, label=label)

    ax.set_title("Arrival Rate vs. Average Delay", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Arrival Rate (lambda)", fontsize=11, labelpad=10)
    ax.set_ylabel("Average Delay (W)", fontsize=11, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="whitesmoke", edgecolor="none")
    plt.tight_layout()
    delay_path = os.path.abspath(os.path.join(output_dir, "arrival_vs_delay.png"))
    plt.savefig(delay_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(delay_path)

    # 2. Arrival Rate vs Drop Probability
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, sweep), color in zip(sweep_dict.items(), colors):
        lambdas = [pt.lambda_rate for pt in sweep.points]
        drops = [pt.drop_probability for pt in sweep.points]
        ax.plot(lambdas, drops, marker="s", linewidth=2, color=color, label=label)

    ax.set_title("Arrival Rate vs. Drop Probability", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Arrival Rate (lambda)", fontsize=11, labelpad=10)
    ax.set_ylabel("Drop Probability", fontsize=11, labelpad=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="whitesmoke", edgecolor="none")
    plt.tight_layout()
    drop_path = os.path.abspath(os.path.join(output_dir, "arrival_vs_drop_prob.png"))
    plt.savefig(drop_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(drop_path)

    # 3. Arrival Rate vs Server Utilization
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, sweep), color in zip(sweep_dict.items(), colors):
        lambdas = [pt.lambda_rate for pt in sweep.points]
        utils = [pt.server_utilization for pt in sweep.points]
        ax.plot(lambdas, utils, marker="^", linewidth=2, color=color, label=label)

    ax.set_title("Arrival Rate vs. Server Utilization", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Arrival Rate (lambda)", fontsize=11, labelpad=10)
    ax.set_ylabel("Server Utilization", fontsize=11, labelpad=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="whitesmoke", edgecolor="none")
    plt.tight_layout()
    util_path = os.path.abspath(os.path.join(output_dir, "arrival_vs_utilization.png"))
    plt.savefig(util_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(util_path)

    return generated_paths
