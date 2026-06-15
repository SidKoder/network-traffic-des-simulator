"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Any

import yaml

from config.models import SimulationConfig


def load_config_from_dict(data: dict[str, Any]) -> SimulationConfig:
    """Build a validated configuration from a dictionary.

    Parameters:
        data: Raw configuration mapping.

    Returns:
        Validated SimulationConfig instance.
    """
    return SimulationConfig.model_validate(data)


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate configuration from a YAML or JSON file.

    Parameters:
        path: Path to a ``.yaml``, ``.yml``, or ``.json`` configuration file.

    Returns:
        Validated SimulationConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file extension is unsupported.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw_text)
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")

    return load_config_from_dict(data)
