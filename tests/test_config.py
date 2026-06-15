"""Tests for configuration models and loader."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.loader import load_config, load_config_from_dict
from config.models import SimulationConfig


def test_simulation_config_valid() -> None:
    """Valid configuration passes validation."""
    config = SimulationConfig(
        arrival={"arrival_rate": 5.0},
        service={"service_rate": 8.0},
        queue={"capacity": None},
        simulation_time=100.0,
        random_seed=42,
    )
    assert config.arrival.arrival_rate == 5.0
    assert config.traffic_intensity == pytest.approx(5.0 / 8.0)
    assert config.is_stable is True


def test_simulation_config_unstable_mm1() -> None:
    """Unstable M/M/1 (rho >= 1) is flagged."""
    config = SimulationConfig(
        arrival={"arrival_rate": 10.0},
        service={"service_rate": 8.0},
        queue={"capacity": None},
        simulation_time=100.0,
    )
    assert config.is_stable is False


def test_simulation_config_mm1k_always_stable_flag() -> None:
    """Finite queue systems are always marked stable."""
    config = SimulationConfig(
        arrival={"arrival_rate": 10.0},
        service={"service_rate": 8.0},
        queue={"capacity": 5},
        simulation_time=100.0,
    )
    assert config.is_stable is True


def test_invalid_arrival_rate_rejected() -> None:
    """Non-positive arrival rate raises ValidationError."""
    with pytest.raises(ValidationError):
        SimulationConfig(
            arrival={"arrival_rate": 0.0},
            service={"service_rate": 8.0},
            queue={"capacity": None},
            simulation_time=100.0,
        )


def test_load_config_from_dict() -> None:
    """Dictionary loader produces valid config."""
    data = {
        "arrival": {"arrival_rate": 3.0},
        "service": {"service_rate": 6.0},
        "queue": {"capacity": 10, "queue_discipline": "FIFO"},
        "simulation_time": 50.0,
    }
    config = load_config_from_dict(data)
    assert config.queue.capacity == 10


def test_load_config_yaml(tmp_path: Path) -> None:
    """YAML file loader works correctly."""
    yaml_content = """
arrival:
  arrival_rate: 4.0
service:
  service_rate: 7.0
queue:
  capacity: null
simulation_time: 80.0
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = load_config(config_file)
    assert config.arrival.arrival_rate == 4.0


def test_load_config_file_not_found() -> None:
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_load_example_configs() -> None:
    """Bundled example configs load without error."""
    root = Path(__file__).parent.parent
    load_config(root / "configs" / "mm1_example.yaml")
    load_config(root / "configs" / "mm1k_example.yaml")
