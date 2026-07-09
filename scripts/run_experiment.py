#!/usr/bin/env python3
"""CLI wrapper to run structured research experiments from configurations."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import yaml

from validation.experiment_runner import ExperimentConfig, ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run discrete-event queueing simulation experiments."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment configuration YAML/JSON file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: configuration file '{args.config}' not found.")
        sys.exit(1)

    print(f"Loading experiment configuration from '{args.config}'...")
    with open(args.config, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    try:
        config = ExperimentConfig.model_validate(raw_data)
    except Exception as e:
        print(f"Error validating experiment configuration:\n{e}")
        sys.exit(1)

    print(f"Starting experiment: '{config.name}' (type: {config.type.value})...")
    runner = ExperimentRunner(config)
    results = runner.execute()

    print("\n" + "=" * 60)
    print(f"Experiment '{config.name}' completed successfully!")
    print(f"Results stored under: {runner.output_dir.resolve()}")
    print("Files created:")
    print(f"  - config.yaml")
    print(f"  - results.json")
    print(f"  - report.md")
    if os.path.exists(runner.output_dir / "plots"):
        plots = os.listdir(runner.output_dir / "plots")
        if plots:
            print(f"  - plots/ ({len(plots)} charts generated)")
    print("=" * 60)


if __name__ == "__main__":
    main()
