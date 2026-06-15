# DES Engine — Probability-Based Packet Arrival Simulator

A production-grade **Discrete Event Simulation (DES)** engine for modeling network packet arrivals, queueing systems, and congestion behavior.

## Features

- **Event-driven architecture** — simulation time advances only through scheduled events
- **Poisson packet arrivals** — homogeneous Poisson process with exponential inter-arrival times
- **Queueing models** — M/M/1 (infinite buffer) and M/M/1/K (finite buffer with drops)
- **Distribution engine** — exponential, normal, gamma, Bernoulli, geometric, weighted discrete
- **Configuration-driven** — no hardcoded simulation parameters
- **Statistically validated** — unit tests verify sample moments against theory

## Project Structure

```
src/
  config/         # Pydantic configuration models and loaders
  distributions/  # Probability distributions (network-agnostic)
  events/         # Event types and priority-queue scheduler
  simulation/     # Clock and packet domain models
  queueing/       # Queue manager (M/M/1, M/M/1/K)
  metrics/        # Performance metrics collection (future)
  analytics/      # Statistical analysis (future)
  utils/          # Shared utilities
tests/            # pytest test suite
configs/          # Example YAML/JSON configuration files
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest -v
```

## Example Configuration

```yaml
# configs/mm1_example.yaml
arrival:
  arrival_rate: 5.0
service:
  service_rate: 8.0
queue:
  capacity: null          # null = infinite (M/M/1)
  queue_discipline: FIFO
simulation_time: 100.0
random_seed: 42
```

```python
from config.loader import load_config
from config.models import SimulationConfig

config = load_config("configs/mm1_example.yaml")
```

## Architecture Principles

1. No hardcoded simulation parameters
2. Strict separation of concerns across modules
3. Dependency injection for testability
4. Composition over inheritance
5. Every module independently testable

## License

MIT
