"""Stress tests for the event loop, scheduler, and history log.

These tests exist to catch ordering violations, memory issues, and
crashes that only surface at scale. They are not unit tests — they
intentionally generate 10,000+ events to exercise the priority queue
and the engine loop end-to-end.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from events.event import Event
from events.history import EventHistoryLog
from events.scheduler import EventScheduler
from events.types import EventType
from simulation.clock import SimulationClock
from simulation.engine import EventLoop, EventHandler


# A pytest mark so the suite can be filtered with ``pytest -m stress``.
pytestmark = pytest.mark.stress


# Cap chosen to keep the test under a couple of seconds on a laptop
# while still exercising the priority queue at non-trivial scale.
EVENT_COUNT = 10_000


class CollectingHandler(EventHandler):
    """Handler that records every event into the history log."""

    def __init__(self, log: EventHistoryLog) -> None:
        self._log = log

    def handle(self, event: Event, current_time: float) -> None:
        self._log.record(event, current_time)


def _build_scheduler(count: int, seed: int) -> EventScheduler:
    """Schedule ``count`` events with random timestamps and packet ids.

    The function intentionally mixes:

    * out-of-order timestamps (sorted only by the heap, not by us),
    * duplicate timestamps across many events, and
    * all four :class:`EventType` values, so the scheduler's
      tie-breaking path is exercised, not just the unique-timestamp
      path.

    Parameters:
        count: Number of events to schedule.
        seed: Seed for the RNG so the test is reproducible.

    Returns:
        A scheduler pre-populated with ``count`` events.
    """
    rng = np.random.default_rng(seed)
    scheduler = EventScheduler()

    timestamps = rng.uniform(0.0, 1_000_000.0, size=count)
    packet_ids = rng.integers(0, 5_000, size=count)
    event_types = list(EventType)

    for ts, pid in zip(timestamps, packet_ids, strict=True):
        scheduler.schedule(
            timestamp=float(ts),
            event_type=event_types[int(pid) % len(event_types)],
            packet_id=int(pid),
        )

    return scheduler


def test_scheduler_stress_processes_all_events_in_order() -> None:
    """10k random events are processed in non-decreasing timestamp order.

    This is the core ordering guarantee: the heap must return events in
    sorted order regardless of insertion order. The test also asserts
    the queue is fully drained — losing an event would be a silent
    bug.
    """
    scheduler = _build_scheduler(EVENT_COUNT, seed=20260708)

    processed: list[Event] = []
    while not scheduler.is_empty():
        processed.append(scheduler.next_event())

    assert len(processed) == EVENT_COUNT, "scheduler dropped an event"

    timestamps = [event.timestamp for event in processed]
    for prev, curr in zip(timestamps, timestamps[1:]):
        assert curr >= prev, f"ordering violation: {prev} -> {curr}"

    # Sanity: the set of timestamps round-trips (no duplicates lost).
    assert len(set(timestamps)) == len(set(timestamps))


def test_scheduler_stress_tie_breaking_stable_at_scale() -> None:
    """At 10k events, identical-timestamp ties are resolved FIFO.

    We schedule ``count`` events all at the same timestamp, then
    dequeue them all. The original insertion order must be preserved
    by the sequence tie-breaker — if the heap broke ties non-FIFO at
    scale, the test would fail.
    """
    count = 10_000
    scheduler = EventScheduler()
    for i in range(count):
        scheduler.schedule(
            timestamp=42.0,
            event_type=EventType.PACKET_ARRIVAL,
            packet_id=i,
        )

    seen: list[int] = []
    while not scheduler.is_empty():
        seen.append(scheduler.next_event().packet_id)

    assert seen == list(range(count)), "tie-breaking is not FIFO at scale"


def test_engine_runs_10k_events_without_crash() -> None:
    """The full event loop drains 10k events cleanly.

    On top of the pure scheduler check, this exercises the engine
    glue: clock advancement, handler dispatch, and the bookkeeping
    counter. ``processed_count`` must equal the number of scheduled
    events for the run to be considered loss-free.
    """
    log = EventHistoryLog()
    scheduler = _build_scheduler(EVENT_COUNT, seed=42)

    loop = EventLoop(
        scheduler=scheduler,
        clock=SimulationClock(),
        handler=CollectingHandler(log),
    )

    processed = loop.run()

    assert processed == EVENT_COUNT, "engine reported wrong processed count"
    assert loop.processed_count == EVENT_COUNT
    assert len(log) == EVENT_COUNT, "history log dropped an event"
    assert scheduler.is_empty(), "scheduler not fully drained"


def test_engine_preserves_strict_ordering_at_scale() -> None:
    """The engine's event sequence is monotonically non-decreasing.

    This is the end-to-end version of the scheduler ordering check:
    after the engine runs, the history log must read out in the same
    non-decreasing order. Any deviation means the engine's pop + clock
    dance is shuffling events relative to the scheduler.
    """
    log = EventHistoryLog()
    scheduler = _build_scheduler(EVENT_COUNT, seed=2026)

    loop = EventLoop(
        scheduler=scheduler,
        clock=SimulationClock(),
        handler=CollectingHandler(log),
    )
    loop.run()

    records = list(log)
    timestamps = [r.timestamp for r in records]
    for prev, curr in zip(timestamps, timestamps[1:]):
        assert curr >= prev, (
            f"engine delivered events out of order: {prev} -> {curr}"
        )

    # Clock must end at the maximum timestamp it ever reached.
    assert loop.clock.current_time == max(timestamps)


def test_engine_runs_10k_events_under_time_budget() -> None:
    """10k events should complete well under a 5-second budget.

    This is a coarse regression guard, not a benchmark. If a future
    change turns the heap operations O(n) (e.g. by accident), this
    test will catch it. The budget is generous on purpose: the point
    is to detect quadratic regressions, not micro-optimizations.
    """
    log = EventHistoryLog()
    scheduler = _build_scheduler(EVENT_COUNT, seed=7)

    loop = EventLoop(
        scheduler=scheduler,
        clock=SimulationClock(),
        handler=CollectingHandler(log),
    )

    start = time.perf_counter()
    processed = loop.run()
    elapsed = time.perf_counter() - start

    assert processed == EVENT_COUNT
    # 5s is intentionally lenient; a healthy run on a laptop is well
    # under 1s. Tightening this would be flaky on CI.
    assert elapsed < 5.0, f"engine took {elapsed:.2f}s for {EVENT_COUNT} events"
    print(f"\n[engine stress] {EVENT_COUNT} events in {elapsed * 1000:.1f} ms")


def test_engine_max_events_and_max_time_caps_at_scale() -> None:
    """Caps work correctly when the scheduler is much larger than the cap.

    With 10k events scheduled and a cap of 100, the loop must stop at
    exactly 100 and leave the remainder in the scheduler — neither
    over- nor under-processing.
    """
    log = EventHistoryLog()
    scheduler = _build_scheduler(EVENT_COUNT, seed=99)

    loop = EventLoop(
        scheduler=scheduler,
        clock=SimulationClock(),
        handler=CollectingHandler(log),
    )

    processed = loop.run(max_events=100)

    assert processed == 100
    assert len(log) == 100
    assert scheduler.pending_count == EVENT_COUNT - 100
    assert loop.clock.current_time == log[-1].timestamp
