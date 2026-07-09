"""Tests for the MetricsCollector.

The collector is a passive storage object — its contract is to record
events, not to compute anything. These tests cover:

* Unit tests of the storage surface (record, filter, slice, export, clear).
* A guardrail test that asserts the collector does not expose any
  computed-method surface (averages, rates, summaries, etc.).
* Integration tests that wire the collector into the project's real
  ``EventLoop`` via a small ``EventHandler`` subclass and assert that
  the recorded sequence matches the expected event types in order.

The integration test copies two test doubles from
``test_integration_simulation.py`` (``ConstantServiceTime``,
``_AlwaysAcceptRng``) so this file is self-contained.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from config.models import QueueConfig, QueueDiscipline
from events.event import Event
from events.scheduler import EventScheduler
from events.types import EventType
from metrics import MetricKind, MetricRecord, MetricsCollector
from queueing.manager import QueueManager
from simulation.engine import EventHandler, EventLoop
from simulation.router import Router
from simulation.server import Server


# ---------------------------------------------------------------------------
# Test doubles (copied from test_integration_simulation.py for self-containment)
# ---------------------------------------------------------------------------


class ConstantServiceTime:
    """Deterministic service-time distribution: every packet takes exactly ``value`` time."""

    def __init__(self, value: float) -> None:
        if value <= 0:
            raise ValueError("service time must be positive")
        self._value = value

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(size, self._value)


class _AlwaysAcceptRng:
    """RNG stand-in whose Bernoulli draw is always 0 (no baseline drop)."""

    def binomial(self, n: int, p: float, size: int) -> np.ndarray:
        return np.zeros(size, dtype=float)


class _DeterministicInterArrival:
    """Poisson-arrival stand-in that consumes a pre-baked list of inter-arrival times.

    Past the end of the list it returns a large sentinel so the in-flight
    packets drain to completion naturally.
    """

    def __init__(self, deltas: list[float]) -> None:
        if not deltas:
            raise ValueError("at least one inter-arrival time is required")
        for d in deltas:
            if d < 0:
                raise ValueError("inter-arrival deltas must be non-negative")
        self._deltas = list(deltas)
        self._index = 0

    def sample_inter_arrival_time(self) -> float:
        if self._index >= len(self._deltas):
            return 10_000.0
        value = self._deltas[self._index]
        self._index += 1
        return float(value)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestEmptyCollector:
    """A freshly constructed collector reports empty state."""

    def test_empty_collector_has_zero_length_and_no_records(self) -> None:
        """A fresh collector has length 0, no records, and empty exports."""
        collector = MetricsCollector()

        assert len(collector) == 0
        assert list(collector) == []
        assert collector.records == []
        assert collector.to_dicts() == []
        assert collector.to_json() == "[]"

    def test_iteration_yields_nothing_when_empty(self) -> None:
        """Iterating an empty collector yields nothing."""
        assert [record for record in MetricsCollector()] == []


class TestRecording:
    """Each ``record_*`` method appends a record with the right shape."""

    def test_record_arrival_appends_record_with_correct_shape(self) -> None:
        """``record_arrival`` stores an ARRIVAL record with the given fields."""
        collector = MetricsCollector()

        collector.record_arrival(0.5, packet_id=1, details={"source": "test"})

        assert len(collector) == 1
        record = collector[0]
        assert record.timestamp == 0.5
        assert record.kind is MetricKind.ARRIVAL
        assert record.packet_id == 1
        assert record.details == {"source": "test"}

    def test_record_service_start_appends_record_with_correct_shape(self) -> None:
        """``record_service_start`` stores a SERVICE_START record with the given fields."""
        collector = MetricsCollector()

        collector.record_service_start(1.0, packet_id=2, details={"queue": "primary"})

        assert len(collector) == 1
        record = collector[0]
        assert record.timestamp == 1.0
        assert record.kind is MetricKind.SERVICE_START
        assert record.packet_id == 2
        assert record.details == {"queue": "primary"}

    def test_record_departure_appends_record_with_correct_shape(self) -> None:
        """``record_departure`` stores a DEPARTURE record with the given fields."""
        collector = MetricsCollector()

        collector.record_departure(
            3.5, packet_id=3, details={"service_time": 2.5}
        )

        assert len(collector) == 1
        record = collector[0]
        assert record.timestamp == 3.5
        assert record.kind is MetricKind.DEPARTURE
        assert record.packet_id == 3
        assert record.details == {"service_time": 2.5}

    def test_record_drop_appends_record_with_correct_shape(self) -> None:
        """``record_drop`` stores a DROP record with the given fields."""
        collector = MetricsCollector()

        collector.record_drop(2.0, packet_id=4, details={"reason": "queue_full"})

        assert len(collector) == 1
        record = collector[0]
        assert record.timestamp == 2.0
        assert record.kind is MetricKind.DROP
        assert record.packet_id == 4
        assert record.details == {"reason": "queue_full"}

    def test_default_details_is_empty_dict_when_omitted(self) -> None:
        """Omitting ``details`` stores ``{}`` on the record."""
        collector = MetricsCollector()

        collector.record_arrival(0.0, packet_id=1)
        collector.record_service_start(0.0, packet_id=1)
        collector.record_departure(0.0, packet_id=1)
        collector.record_drop(0.0, packet_id=1)

        assert all(record.details == {} for record in collector)

    def test_details_are_stored_as_defensive_copy(self) -> None:
        """Mutating the caller's input dict after recording does not affect the record."""
        collector = MetricsCollector()
        details = {"reason": "queue_full"}

        collector.record_drop(0.0, packet_id=1, details=details)
        details["reason"] = "tampered"

        assert collector[0].details == {"reason": "queue_full"}

    def test_records_property_returns_a_copy(self) -> None:
        """Mutating the ``records`` snapshot does not affect the collector."""
        collector = MetricsCollector()
        collector.record_arrival(0.0, packet_id=1)
        collector.record_arrival(1.0, packet_id=2)

        snapshot = collector.records
        snapshot.append("not a record")  # type: ignore[arg-type]

        assert len(collector) == 2


class TestFilter:
    """``filter_by_kind`` and ``filter_by_packet`` return matching records in order."""

    def test_filter_by_kind_returns_only_matching_records_in_insertion_order(self) -> None:
        """filter_by_kind preserves insertion order across multiple kinds."""
        collector = MetricsCollector()
        collector.record_arrival(0.0, packet_id=1)
        collector.record_drop(0.1, packet_id=2, details={"reason": "baseline_drop"})
        collector.record_service_start(0.0, packet_id=1)
        collector.record_drop(0.2, packet_id=3, details={"reason": "queue_full"})

        drops = collector.filter_by_kind(MetricKind.DROP)
        assert [record.packet_id for record in drops] == [2, 3]
        assert [record.details["reason"] for record in drops] == [
            "baseline_drop",
            "queue_full",
        ]

        assert len(collector.filter_by_kind(MetricKind.ARRIVAL)) == 1
        assert len(collector.filter_by_kind(MetricKind.SERVICE_START)) == 1
        assert len(collector.filter_by_kind(MetricKind.DEPARTURE)) == 0

    def test_filter_by_packet_returns_only_matching_records_in_insertion_order(self) -> None:
        """filter_by_packet spans kinds and preserves insertion order."""
        collector = MetricsCollector()
        collector.record_arrival(0.0, packet_id=1)
        collector.record_service_start(0.0, packet_id=1)
        collector.record_departure(2.0, packet_id=1)
        collector.record_arrival(1.0, packet_id=2)
        collector.record_drop(1.0, packet_id=2, details={"reason": "queue_full"})

        packet_one = collector.filter_by_packet(1)
        assert [record.kind for record in packet_one] == [
            MetricKind.ARRIVAL,
            MetricKind.SERVICE_START,
            MetricKind.DEPARTURE,
        ]

        packet_two = collector.filter_by_packet(2)
        assert [record.kind for record in packet_two] == [
            MetricKind.ARRIVAL,
            MetricKind.DROP,
        ]


class TestIndexing:
    """``__getitem__`` supports both integer and slice indexing."""

    def test_getitem_by_index_and_slice(self) -> None:
        """Indexing returns a single record; slicing returns a list."""
        collector = MetricsCollector()
        for packet_id in range(1, 5):
            collector.record_arrival(float(packet_id), packet_id=packet_id)

        assert isinstance(collector[0], MetricRecord)
        assert collector[0].packet_id == 1
        assert collector[-1].packet_id == 4

        sliced = collector[1:3]
        assert isinstance(sliced, list)
        assert [record.packet_id for record in sliced] == [2, 3]

    def test_getitem_out_of_range_raises(self) -> None:
        """IndexError on out-of-range integer index."""
        collector = MetricsCollector()
        collector.record_arrival(0.0, packet_id=1)

        with pytest.raises(IndexError):
            _ = collector[5]


class TestClear:
    """``clear`` empties the collector."""

    def test_clear_empties_the_collector(self) -> None:
        """After clear, the collector reports empty state and re-accepts records."""
        collector = MetricsCollector()
        collector.record_arrival(0.0, packet_id=1)
        collector.record_departure(1.0, packet_id=1)

        collector.clear()

        assert len(collector) == 0
        assert collector.records == []
        assert collector.to_json() == "[]"

        # The collector is still usable after clear.
        collector.record_arrival(2.0, packet_id=2)
        assert len(collector) == 1


class TestExport:
    """``to_dicts`` and ``to_json`` produce JSON-serializable output."""

    def test_to_dicts_renders_kind_as_string(self) -> None:
        """to_dicts renders the ``kind`` enum as its string value."""
        collector = MetricsCollector()
        collector.record_drop(0.0, packet_id=1, details={"reason": "queue_full"})

        rendered = collector.to_dicts()
        assert len(rendered) == 1
        assert rendered[0]["kind"] == "drop"
        assert rendered[0]["packet_id"] == 1
        assert rendered[0]["details"] == {"reason": "queue_full"}

    def test_to_json_round_trips_via_json_module(self) -> None:
        """to_json output is a valid JSON array that round-trips through json.loads."""
        collector = MetricsCollector()
        collector.record_arrival(0.0, packet_id=1)
        collector.record_service_start(0.0, packet_id=1)
        collector.record_departure(2.0, packet_id=1, details={"service_time": 2.0})
        collector.record_drop(1.0, packet_id=2, details={"reason": "queue_full"})

        payload = collector.to_json()
        assert isinstance(payload, str)

        decoded = json.loads(payload)
        assert decoded == collector.to_dicts()
        assert len(decoded) == 4


class TestRecordFrozen:
    """``MetricRecord`` is frozen and cannot be mutated after creation."""

    def test_metric_record_is_frozen(self) -> None:
        """Attribute assignment on a MetricRecord raises FrozenInstanceError."""
        record = MetricRecord(
            timestamp=0.0, kind=MetricKind.ARRIVAL, packet_id=1
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            record.timestamp = 1.0  # type: ignore[misc]


class TestNoComputationGuard:
    """The collector must not expose any computed-method surface.

    This guardrail guards against accidental drift toward computation.
    The collector's contract is to store, not to compute. If a future
    change adds any of the names below, this test will fail and the
    contributor will be forced to justify the change.
    """

    def test_collector_does_not_expose_computed_methods(self) -> None:
        """No forbidden names are present on a MetricsCollector instance."""
        forbidden = (
            "average",
            "mean",
            "sum",
            "count_drops",
            "count_packets",
            "drop_rate",
            "throughput",
            "utilization",
            "summary",
            "stats",
            "report",
        )
        collector = MetricsCollector()

        for name in forbidden:
            assert not hasattr(collector, name), (
                f"MetricsCollector exposes a computed attribute/method "
                f"named {name!r}; the collector must not compute."
            )


# ---------------------------------------------------------------------------
# Integration tests — wire the collector into the real engine.
# ---------------------------------------------------------------------------


class _CollectingHandler(EventHandler):
    """Dispatch every event to the router and record it in the collector.

    The handler mirrors the project convention from
    ``test_integration_simulation._WiringHandler``: the router is called
    first so that ``PACKET_ARRIVAL`` events get their packet id assigned
    *before* the arrival is recorded.
    """

    def __init__(
        self,
        router: Router,
        collector: MetricsCollector,
        scheduler: EventScheduler,
        arrival_proc: _DeterministicInterArrival,
    ) -> None:
        self._router = router
        self._collector = collector
        self._scheduler = scheduler
        self._arrival_proc = arrival_proc

    def handle(self, event: Event, current_time: float):
        if event.event_type == EventType.PACKET_ARRIVAL:
            packet = self._router.handle_event(event)
            self._collector.record_arrival(current_time, packet.packet_id)

            next_dt = self._arrival_proc.sample_inter_arrival_time()
            next_ts = current_time + next_dt
            if next_ts < 10_000.0:
                self._scheduler.schedule(
                    next_ts, EventType.PACKET_ARRIVAL, packet_id=None
                )
            return None

        if event.event_type == EventType.PACKET_SERVICE_START:
            packet = self._router.handle_event(event)
            self._collector.record_service_start(
                current_time, packet.packet_id, details=event.metadata
            )
            return None

        if event.event_type == EventType.PACKET_DEPARTURE:
            packet = self._router.handle_event(event)
            self._collector.record_departure(
                current_time,
                packet.packet_id,
                details={"service_time": event.metadata.get("service_time")},
            )
            return None

        if event.event_type == EventType.PACKET_DROP:
            self._router.handle_event(event)
            drop_packet = event.metadata.get("packet") if event.metadata else None
            packet_id = (
                drop_packet.packet_id
                if drop_packet is not None
                else event.packet_id
            )
            self._collector.record_drop(
                current_time,
                packet_id,
                details={"reason": event.metadata.get("reason") if event.metadata else None},
            )
            return None

        return None

    def on_start(self, current_time: float) -> None:
        first_dt = self._arrival_proc.sample_inter_arrival_time()
        first_ts = current_time + first_dt
        if first_ts < 10_000.0:
            self._scheduler.schedule(
                first_ts, EventType.PACKET_ARRIVAL, packet_id=None
            )


def _build_simulator(
    *,
    capacity: int | None,
    inter_arrivals: list[float],
    service_time: float = 1.0,
) -> tuple[
    Router, EventScheduler, MetricsCollector, EventLoop, _CollectingHandler
]:
    """Construct a fully wired DES run with the collector wired in."""
    queue = QueueManager(
        QueueConfig(capacity=capacity, queue_discipline=QueueDiscipline.FIFO)
    )
    server = Server()
    scheduler = EventScheduler()
    service_dist = ConstantServiceTime(service_time)
    arrival_proc = _DeterministicInterArrival(inter_arrivals)
    collector = MetricsCollector()

    router = Router(
        queue_manager=queue,
        server=server,
        scheduler=scheduler,
        service_time_distribution=service_dist,
        baseline_drop_probability=0.0,
        rng=_AlwaysAcceptRng(),  # type: ignore[arg-type]
    )

    handler = _CollectingHandler(router, collector, scheduler, arrival_proc)
    loop = EventLoop(scheduler=scheduler, handler=handler)
    return router, scheduler, collector, loop, handler


class TestIntegration:
    """End-to-end tests of the collector wired into the engine."""

    def test_integration_single_packet_records_full_lifecycle(self) -> None:
        """One packet: ARRIVAL -> SERVICE_START -> DEPARTURE, recorded in order."""
        _, _, collector, loop, _ = _build_simulator(
            capacity=None,
            inter_arrivals=[0.0, 10_000.0],
            service_time=2.0,
        )

        loop.run()

        assert len(collector) == 3
        assert [record.kind for record in collector] == [
            MetricKind.ARRIVAL,
            MetricKind.SERVICE_START,
            MetricKind.DEPARTURE,
        ]
        assert all(record.packet_id == 1 for record in collector)
        assert [record.timestamp for record in collector] == [0.0, 0.0, 2.0]
        assert collector[-1].details == {"service_time": 2.0}

    def test_integration_records_drop_event_with_reason_for_queue_full_packet(self) -> None:
        """Queue-full drops are recorded with the ``queue_full`` reason."""
        # capacity=1, service time=2.0:
        #   t=0.0  packet 1 arrives, server idle -> service starts immediately
        #   t=0.5  packet 2 arrives, server busy -> queued
        #   t=1.0  packet 3 arrives, server busy + queue full -> dropped
        #   t=1.5  packet 4 arrives, server busy + queue full -> dropped
        #   sentinel at t=10_000 ends the arrival stream; in-flight packets
        #   drain to completion (service end at t=2.0 and t=4.0).
        _, _, collector, loop, _ = _build_simulator(
            capacity=1,
            inter_arrivals=[0.0, 0.5, 0.5, 0.5, 10_000.0],
            service_time=2.0,
        )

        loop.run()

        drops = collector.filter_by_kind(MetricKind.DROP)
        assert len(drops) == 2
        assert [record.packet_id for record in drops] == [3, 4]
        assert all(record.details == {"reason": "queue_full"} for record in drops)
        assert [record.timestamp for record in drops] == [1.0, 1.5]

        # Kinds, in recording order:
        #   arrival(1), service_start(1), arrival(2), arrival(3) + drop(3),
        #   arrival(4) + drop(4), departure(1), service_start(2), departure(2).
        assert [record.kind for record in collector] == [
            MetricKind.ARRIVAL,        # packet 1 at t=0.0
            MetricKind.SERVICE_START,  # packet 1 at t=0.0
            MetricKind.ARRIVAL,        # packet 2 at t=0.5
            MetricKind.ARRIVAL,        # packet 3 at t=1.0
            MetricKind.DROP,           # packet 3 at t=1.0
            MetricKind.ARRIVAL,        # packet 4 at t=1.5
            MetricKind.DROP,           # packet 4 at t=1.5
            MetricKind.DEPARTURE,      # packet 1 at t=2.0
            MetricKind.SERVICE_START,  # packet 2 at t=2.0
            MetricKind.DEPARTURE,      # packet 2 at t=4.0
        ]
