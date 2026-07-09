# DES Engine — Probability-Based Packet Arrival Simulator

A production-grade **Discrete Event Simulation (DES)** engine for modeling network packet arrivals, queueing systems, and congestion behavior.

## Features

- **Event-driven architecture** — simulation time advances only through scheduled events
- **Priority-queue scheduler** — heap-backed `EventScheduler` with FIFO tie-breaking for equal timestamps
- **Generic event loop** — `EventLoop` coordinates the clock, scheduler, and a pluggable `EventHandler`
- **Pluggable event handlers** — domain logic lives outside the engine via the `EventHandler` interface; default `PrintEventHandler` ships for smoke tests
- **Event history log** — append-only `EventHistoryLog` for post-mortem debugging (timestamp, type, packet, event id)
- **Simulation clock** — `SimulationClock` with strict forward-only time movement (rejects backward jumps)
- **Poisson packet arrivals** — homogeneous Poisson process with exponential inter-arrival times
- **Queueing models** — M/M/1 (infinite buffer) and M/M/1/K (finite buffer with drops)
- **Distribution engine** — exponential, normal, gamma, Bernoulli, geometric, weighted discrete
- **Configuration-driven** — no hardcoded simulation parameters
- **Statistically validated** — unit tests verify sample moments against theory
- **Stress tested** — 10,000-event end-to-end runs verify ordering, runtime, and absence of crashes

## Project Structure

```
src/
  config/         # Pydantic configuration models and loaders
  distributions/  # Probability distributions (network-agnostic)
  events/         # Event types, priority-queue scheduler, history log
  simulation/     # Clock, generic event loop, event handler interface, packet model
  queueing/       # Queue manager (M/M/1, M/M/1/K)
  metrics/        # Performance metrics collection (future)
  analytics/      # Statistical analysis helpers (packet-level latency stats)
  utils/          # Shared utilities
tests/            # pytest test suite (unit + stress)
configs/          # Example YAML/JSON configuration files
```

## Core Abstractions

The engine is built from four cooperating components. Each is independently importable and testable.

### `Event` and `EventType`

`Event` is the unit of work carried by the scheduler. It compares chronologically by `(timestamp, sequence)` so the heap can break ties deterministically. All descriptive fields (`event_type`, `packet_id`, `metadata`, `event_id`) are excluded from comparisons so they never affect ordering.

```python
from events import Event, EventType

event = Event(
    timestamp=1.5,
    event_type=EventType.PACKET_ARRIVAL,
    packet_id=42,
    metadata={"source": "generator"},
)
```

`EventType` currently exposes four actions: `PACKET_ARRIVAL`, `PACKET_SERVICE_START`, `PACKET_DEPARTURE`, `PACKET_DROP`. The set is intentionally small — domain-specific actions should be modeled as `EventType` values plus metadata, not as new event classes.

### `EventScheduler` — priority-queue scheduler

`EventScheduler` is a min-heap of `Event` objects. It assigns a monotonically increasing `sequence` number on insert so events with identical timestamps are returned in insertion order (FIFO). The heap operations are O(log n) per insert/remove.

```python
from events import EventScheduler, EventType

scheduler = EventScheduler()
scheduler.schedule(5.0, EventType.PACKET_DEPARTURE, packet_id=3)
scheduler.schedule(1.0, EventType.PACKET_ARRIVAL, packet_id=1)
scheduler.schedule(3.0, EventType.PACKET_SERVICE_START, packet_id=2)

while not scheduler.is_empty():
    event = scheduler.next_event()
    # delivered in order: 1.0, 3.0, 5.0
```

Other entry points: `peek()` (inspect the head without popping), `add_event()` (schedule a pre-built `Event`), `clear()` (drain), and the iterator protocol (drains the queue in order).

### `SimulationClock`

`SimulationClock` owns the current simulation time and refuses to move backward. Time advances only when the engine moves it to an event's timestamp — there is no real-time or sleep-based progression, so simulation speed is bounded by engine throughput, not wall clock.

```python
from simulation import SimulationClock

clock = SimulationClock(initial_time=0.0)
clock.advance_to(10.5)   # returns 10.5
clock.advance_to(8.0)    # raises ValueError: Cannot move clock backward
clock.reset(0.0)
```

### `EventHandler` interface and `EventLoop`

`EventLoop` ties the clock and scheduler together with a pluggable `EventHandler`. Each iteration it peeks the next event, checks the optional `max_events` and `max_time` caps, advances the clock, and dispatches the event to the handler. Follow-up events returned by the handler are re-injected into the scheduler, so event chains (e.g. `ARRIVAL` → `SERVICE_START` → `DEPARTURE`) work without any domain logic in the engine itself.

```python
from events import EventScheduler, EventType
from simulation import EventHandler, EventLoop, SimulationClock


class TraceHandler(EventHandler):
    """Custom handler: prints each event and tracks the packet timeline."""

    def __init__(self) -> None:
        self.history: list[tuple[float, str, int | None]] = []

    def handle(self, event, current_time):
        self.history.append((current_time, event.event_type.name, event.packet_id))


loop = EventLoop(
    scheduler=EventScheduler(),
    clock=SimulationClock(),
    handler=TraceHandler(),
)
loop.scheduler.schedule(0.5, EventType.PACKET_ARRIVAL, packet_id=1)
loop.scheduler.schedule(2.0, EventType.PACKET_DEPARTURE, packet_id=1)
loop.scheduler.schedule(1.25, EventType.PACKET_SERVICE_START, packet_id=2)

processed = loop.run()
# processed == 3
# loop.clock.current_time == 2.0
# loop.handler.history == [(0.5, 'PACKET_ARRIVAL', 1),
#                          (1.25, 'PACKET_SERVICE_START', 2),
#                          (2.0, 'PACKET_DEPARTURE', 1)]
```

A default `PrintEventHandler` ships with the engine — it logs each event to standard output (via the `logging` module, or via `print` if `use_logging=False`). It's useful as a smoke test and as a reference for writing concrete handlers.

The loop accepts two optional caps:
- `max_events` — stop after this many events are processed.
- `max_time` — stop when the next event would fire past this simulation time (the current event is *not* processed).

### `EventHistoryLog` — debugging and post-mortem

`EventHistoryLog` is an append-only log of processed events, designed to be wired into the engine as an `EventHandler`. It captures `(timestamp, event_type, packet_id, event_id)` per event and exposes them as frozen `HistoryRecord` objects so the audit trail cannot be silently mutated.

```python
from events import EventHistoryLog
from simulation import EventLoop


log = EventHistoryLog()


class HistoryHandler:
    def handle(self, event, current_time):
        log.record(event, current_time)


loop = EventLoop(handler=HistoryHandler())
# ... schedule and run ...
print(log.to_json())                  # JSON array of records
print(log.filter_by_packet(packet_id)) # events for one packet
print(log.filter_by_type(EventType.PACKET_DEPARTURE))
```

The log has no built-in size cap; call `clear()` between long runs to bound memory. Records are frozen (`@dataclass(frozen=True)`) so a downstream bug cannot rewrite history.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the full test suite (unit + stress)
pytest -v

# Run only the stress tests (10k-event engine runs)
pytest -v -m stress
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
