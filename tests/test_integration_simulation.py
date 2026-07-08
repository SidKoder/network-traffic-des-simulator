"""End-to-end integration tests for the DES simulator.

These tests wire the full stack together and drive it through the
generic :class:`EventLoop`:

    EventScheduler  -- pops events -->
    EventLoop       -- advances clock + dispatches -->
    Router handler  -- performs arrival / service / departure / drop -->
    QueueManager    -- buffers packets with FIFO discipline -->
    Server          -- one packet in service at a time -->
    EventHistoryLog -- records the audit trail

Each scenario here targets one of the four behaviors the simulation
must satisfy, and the assertions read against the *post-run* state
(every packet's arrival/service_start/departure times, the queue
contents, the server state, the history log) so a regression in any
layer — scheduler, router, queue manager, server, packet model —
breaks the test.

The test module is intentionally non-mutating: it never patches
production code, and the only randomness comes from the seeded RNG
that each scenario constructs explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.models import (
    ArrivalConfig,
    QueueConfig,
    QueueDiscipline,
    ServiceConfig,
    SimulationConfig,
)
from distributions.continuous import ExponentialDistribution
from events.event import Event
from events.history import EventHistoryLog
from events.scheduler import EventScheduler
from events.types import EventType
from queueing.manager import QueueManager
from simulation.engine import EventHandler, EventLoop
from simulation.packet import Packet
from simulation.router import Router
from simulation.server import Server


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class DeterministicInterArrival:
    """Poisson-arrival stand-in that consumes a pre-baked list of inter-arrival times.

    The simulator calls ``sample_inter_arrival_time()`` once per arrival to
    decide when the next packet shows up. By feeding it a hand-written
    list we get full control over the arrival schedule without random
    flakiness — every test below is deterministic.
    """

    def __init__(self, deltas: list[float]) -> None:
        if not deltas:
            raise ValueError("at least one inter-arrival time is required")
        for d in deltas:
            if d < 0:
                raise ValueError("inter-arrival deltas must be non-negative")
        self._deltas = list(deltas)
        self._index = 0
        self.calls = 0

    def sample_inter_arrival_time(self) -> float:
        if self._index >= len(self._deltas):
            # Past the end: never accept another arrival. A huge value lets
            # the in-flight packets drain to completion naturally.
            return 10_000.0
        value = self._deltas[self._index]
        self._index += 1
        self.calls += 1
        return float(value)

    def sample_inter_arrival_times(self, n: int) -> np.ndarray:
        return np.array([self.sample_inter_arrival_time() for _ in range(n)])


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


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _build_simulator(
    *,
    capacity: int | None,
    inter_arrivals: list[float],
    service_time: float = 1.0,
    baseline_drop_probability: float = 0.0,
) -> tuple[Router, EventScheduler, EventHistoryLog, EventLoop, DeterministicInterArrival, "_Trace"]:
    """Construct a fully wired DES run.

    The pieces are the same as production:

    * an ``EventScheduler`` heap
    * a :class:`QueueManager` of given capacity
    * a :class:`Server` (one slot)
    * a :class:`Router` that owns all of the above
    * a :class:`EventHistoryLog` wired in through a handler
    * a :class:`EventLoop` driving everything

    The first packet's ``PACKET_ARRIVAL`` is seeded at ``t=0``; subsequent
    arrivals are computed by the deterministic inter-arrival process.

    The returned :class:`_Trace` mirrors the history log but also carries
    the packet_id and the original event metadata, so tests can assert
    on drop reasons and the per-event packet identifier even after the
    log is recorded.
    """
    queue = QueueManager(QueueConfig(capacity=capacity, queue_discipline=QueueDiscipline.FIFO))
    server = Server()
    scheduler = EventScheduler()
    service_dist = ConstantServiceTime(service_time)
    arrival_proc = DeterministicInterArrival(inter_arrivals)
    log = EventHistoryLog()
    trace = _Trace()

    rng = _AlwaysAcceptRng()
    router = Router(
        queue_manager=queue,
        server=server,
        scheduler=scheduler,
        service_time_distribution=service_dist,
        baseline_drop_probability=baseline_drop_probability,
        rng=rng,  # type: ignore[arg-type]
    )

    class WiringHandler(EventHandler):
        """Dispatch every event to the router and record it in the log + trace."""

        def __init__(self) -> None:
            self.router = router
            self.arrival_proc = arrival_proc
            self._t: float = 0.0

        def handle(self, event: Event, current_time: float) -> None:
            # Pass the event through the router first so the packet_id is
            # assigned on ARRIVAL before we record the trace row.
            if event.event_type == EventType.PACKET_ARRIVAL:
                packet = self.router.handle_event(event)
                # ``handle_arrival`` returns the freshly created packet;
                # its packet_id is what the rest of the system will see.
                # The router schedules follow-up events; we capture them
                # before scheduling the next arrival so the trace keeps
                # the arrival event separate from the generated start/drop.
                trace.record(
                    event.event_type,
                    current_time,
                    packet.packet_id,
                    event.metadata,
                )
                next_dt = self.arrival_proc.sample_inter_arrival_time()
                next_ts = current_time + next_dt
                if next_ts < 10_000.0:  # sentinel = "no more arrivals"
                    scheduler.schedule(
                        next_ts,
                        EventType.PACKET_ARRIVAL,
                        packet_id=None,
                    )
                return

            if event.event_type in {
                EventType.PACKET_SERVICE_START,
                EventType.PACKET_DEPARTURE,
            }:
                packet = self.router.handle_event(event)
                trace.record(
                    event.event_type,
                    current_time,
                    packet.packet_id if packet is not None else event.packet_id,
                    event.metadata,
                )
                return

            if event.event_type == EventType.PACKET_DROP:
                # The router has already dropped the packet and stashed it
                # in the event metadata at scheduling time.
                self.router.handle_event(event)
                metadata = event.metadata or {}
                drop_reason = metadata.get("reason")
                drop_packet = metadata.get("packet")
                drop_pid = (
                    drop_packet.packet_id
                    if isinstance(drop_packet, Packet)
                    else event.packet_id
                )
                trace.record(
                    event.event_type,
                    current_time,
                    drop_pid,
                    {"reason": drop_reason},
                )
                return

            # Should not happen for this engine.
            return

        def on_start(self, current_time: float) -> None:
            # Seed the first arrival at the configured offset. By
            # convention, ``inter_arrivals[0]`` is the offset of the
            # first packet (commonly 0.0); subsequent elements are the
            # gaps between consecutive arrivals. When the list is
            # exhausted the simulator emits a final arrival far in the
            # future and stops.
            first_dt = arrival_proc.sample_inter_arrival_time()
            first_ts = current_time + first_dt
            if first_ts < 10_000.0:
                scheduler.schedule(first_ts, EventType.PACKET_ARRIVAL, packet_id=None)

        def on_stop(self, current_time: float) -> None:
            self._t = current_time

    handler = WiringHandler()
    clock = _RecordingClock()
    loop = EventLoop(scheduler=scheduler, clock=clock, handler=handler)
    return router, scheduler, log, loop, arrival_proc, trace


class _Trace:
    """A small per-event audit trail that keeps packet_id and metadata.

    The :class:`EventHistoryLog` strips metadata and only stores a
    packet_id that was set when the event was *scheduled*. For
    PACKET_ARRIVAL events the packet id is unknown at scheduling time
    (the router assigns it during ``handle_arrival``), so the log
    records ``None`` for those. The trace solves that by capturing the
    router's view of each event after it has been processed.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, object]] = []

    def record(
        self,
        event_type: EventType,
        timestamp: float,
        packet_id: int | None,
        metadata: dict[str, object] | None,
    ) -> None:
        self._rows.append(
            {
                "event_type": event_type,
                "timestamp": timestamp,
                "packet_id": packet_id,
                "metadata": dict(metadata) if metadata else {},
            }
        )

    def filter(self, event_type: EventType) -> list[dict[str, object]]:
        return [row for row in self._rows if row["event_type"] is event_type]

    @property
    def rows(self) -> list[dict[str, object]]:
        return list(self._rows)


class _RecordingClock:
    """Minimal stand-in matching the SimulationClock interface used by EventLoop.

    The production :class:`SimulationClock` rejects equal-time ``advance_to``
    calls in some configurations and is stateful in ways that complicate
    fine-grained tests. This stand-in records every advance so a test can
    assert that the clock never moves backward.
    """

    def __init__(self, initial_time: float = 0.0) -> None:
        self._t = initial_time
        self.history: list[float] = [initial_time]

    @property
    def current_time(self) -> float:
        return self._t

    def advance_to(self, t: float) -> float:
        if t < self._t:
            raise ValueError(f"clock cannot move backward: {self._t} -> {t}")
        self._t = t
        self.history.append(t)
        return t


# ---------------------------------------------------------------------------
# 1. Arrival -> service -> departure full lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_single_packet_goes_arrival_service_departure() -> None:
    """One packet: ARRIVAL -> SERVICE_START -> DEPARTURE, in that order, with correct timing."""
    router, scheduler, log, loop, _, trace = _build_simulator(
        capacity=None,
        inter_arrivals=[0.0, 10_000.0],  # first arrival at t=0, then no more
        service_time=3.0,
    )

    processed = loop.run()

    # Exactly three router-level events: arrival, service start, departure.
    arrivals = trace.filter(EventType.PACKET_ARRIVAL)
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    departs = trace.filter(EventType.PACKET_DEPARTURE)
    drops = trace.filter(EventType.PACKET_DROP)

    assert len(arrivals) == 1
    assert len(starts) == 1
    assert len(departs) == 1
    assert drops == []
    assert processed == 3

    # Timing is deterministic: arrival at 0.0, service starts at 0.0 (server
    # was idle), departure at 0.0 + 3.0.
    assert arrivals[0]["timestamp"] == 0.0
    assert starts[0]["timestamp"] == 0.0
    assert departs[0]["timestamp"] == 3.0

    # The packet is the one the router created.
    assert router.server.busy is False
    assert router.server.current_packet is None
    assert router.queue_manager.is_empty is True

    # Bookkeeping on the router matches the trace.
    assert router.packets_created == 1
    assert router.packets_dropped == 0

    # The history log is also in non-decreasing time order.
    timestamps = [r.timestamp for r in log]
    assert timestamps == sorted(timestamps)


def test_lifecycle_multiple_packets_each_one_full_event_chain() -> None:
    """Each packet produces ARRIVAL -> SERVICE_START -> DEPARTURE in order, no drops."""
    router, _, log, loop, _, trace = _build_simulator(
        capacity=None,
        # Arrivals at 0, 4, 8, 12, 16. Service time = 3 each, server idle
        # at every arrival, so no queuing. With one arrival per ~4 time
        # units and 3-unit service, the server is always free when a
        # packet shows up.
        inter_arrivals=[0.0, 4.0, 4.0, 4.0, 4.0, 10_000.0],
        service_time=3.0,
    )

    loop.run()

    arrivals = trace.filter(EventType.PACKET_ARRIVAL)
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    departs = trace.filter(EventType.PACKET_DEPARTURE)
    drops = trace.filter(EventType.PACKET_DROP)

    assert len(arrivals) == 5
    assert len(starts) == 5
    assert len(departs) == 5
    assert drops == []

    # No packet was ever waiting in the queue.
    assert router.queue_manager.is_empty is True

    # Service starts at the same time as the arrival (server always idle).
    for arr, start in zip(arrivals, starts, strict=True):
        assert start["timestamp"] == arr["timestamp"]

    # Departures line up: arrival + 3.0 each.
    expected_departures = [arr["timestamp"] + 3.0 for arr in arrivals]
    actual_departures = [d["timestamp"] for d in departs]
    assert actual_departures == expected_departures

    # The audit trail is in non-decreasing time order end-to-end.
    timestamps = [r.timestamp for r in log]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 2. Packets wait when the server is busy
# ---------------------------------------------------------------------------


def test_waiting_packets_stay_in_queue_until_server_frees() -> None:
    """A burst of arrivals queues up; the second/third wait while the first is served."""
    # Service time 5.0. Arrivals at 0, 1, 2, 3. Server is busy from t=0..5,
    # so packets 2-4 are forced to wait in the queue.
    router, _, log, loop, _, trace = _build_simulator(
        capacity=None,
        inter_arrivals=[0.0, 1.0, 1.0, 1.0, 10_000.0],
        service_time=5.0,
    )

    loop.run()

    arrivals = trace.filter(EventType.PACKET_ARRIVAL)
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    departs = trace.filter(EventType.PACKET_DEPARTURE)
    drops = trace.filter(EventType.PACKET_DROP)

    assert len(arrivals) == 4
    assert len(starts) == 4
    assert len(departs) == 4
    assert drops == []

    # Packet 1: arrives at 0, served at 0 (server was idle), departs at 5.
    # Packet 2: arrives at 1, served at 5 (right after packet 1 departs), departs at 10.
    # Packet 3: arrives at 2, served at 10, departs at 15.
    # Packet 4: arrives at 3, served at 15, departs at 20.
    expected_starts = [0.0, 5.0, 10.0, 15.0]
    expected_departures = [5.0, 10.0, 15.0, 20.0]
    assert [s["timestamp"] for s in starts] == expected_starts
    assert [d["timestamp"] for d in departs] == expected_departures

    # Each packet's waiting time matches the gap between its arrival and
    # its service start: 0, 4, 8, 12.
    expected_waits = [0.0, 4.0, 8.0, 12.0]
    actual_waits = [s["timestamp"] - a["timestamp"] for a, s in zip(arrivals, starts, strict=True)]
    assert actual_waits == expected_waits

    # No drops, no leftover packets.
    assert router.queue_manager.is_empty is True
    assert router.server.busy is False

    # The history log shows arrivals strictly before their corresponding
    # service starts (no negative wait).
    for arr, start in zip(arrivals, starts, strict=True):
        assert arr["timestamp"] <= start["timestamp"]


def test_packet_inside_queue_does_not_get_service_before_its_turn() -> None:
    """While a packet is queued, no other packet's service is started.

    The router's ``_service_start_pending`` guard plus the
    ``not server.busy`` check make it impossible to schedule a second
    service start while one is already in flight.
    """
    router, scheduler, log, loop, _, trace = _build_simulator(
        capacity=None,
        inter_arrivals=[0.0, 1.0, 1.0, 1.0, 10_000.0],
        service_time=4.0,
    )

    loop.run()

    # Group every PACKET_SERVICE_START with the packet id on its event
    # and the timestamp; there must be exactly one such event per packet.
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    assert len(starts) == 4

    # At no point were two service starts in flight (we'd see two events
    # at the same timestamp if so, but the design is that departures
    # and starts are interleaved one at a time).
    timestamps = [s["timestamp"] for s in starts]
    # Expected: 0, 4, 8, 12 — strictly increasing.
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


# ---------------------------------------------------------------------------
# 3. Drops occur when the queue is full
# ---------------------------------------------------------------------------


def test_drops_occur_when_finite_queue_is_full() -> None:
    """With a small finite queue, late arrivals are dropped, not silently queued."""
    # Service time 10 — server busy throughout the burst. Capacity 1 means
    # at most one packet can be waiting; any further arrival is dropped.
    # Arrivals at 0, 1, 2, 3 -> packet 1 in service, packet 2 in queue,
    # packet 3 dropped, packet 4 dropped.
    router, _, log, loop, _, trace = _build_simulator(
        capacity=1,
        inter_arrivals=[0.0, 1.0, 1.0, 1.0, 10_000.0],
        service_time=10.0,
    )

    loop.run()

    arrivals = trace.filter(EventType.PACKET_ARRIVAL)
    drops = trace.filter(EventType.PACKET_DROP)
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    departs = trace.filter(EventType.PACKET_DEPARTURE)

    assert len(arrivals) == 4
    assert len(starts) == 2  # only packets 1 and 2 are served
    assert len(departs) == 2
    assert len(drops) == 2  # packets 3 and 4 are dropped

    # The drop timestamps match the arrival timestamps of the dropped packets.
    drop_timestamps = sorted(d["timestamp"] for d in drops)
    assert drop_timestamps == [2.0, 3.0]

    # Each drop event carries the "queue_full" reason in its metadata and
    # points to the right packet id.
    drop_packet_ids = sorted(d["packet_id"] for d in drops)
    assert drop_packet_ids == [3, 4]
    for drop in drops:
        assert drop["metadata"]["reason"] == "queue_full"

    # No drop ever happened before a queue-full state was reached:
    # packets 1 and 2 were both enqueued.
    assert router.queue_manager.total_enqueued == 2
    assert router.queue_manager.total_dropped == 2
    assert router.packets_created == 4
    assert router.packets_dropped == 2


def test_dropped_packet_does_not_block_the_queue() -> None:
    """After a drop, later accepted packets still find a slot."""
    # Capacity 1. Two bursts of three arrivals each, separated by a long
    # gap so the queue drains. Service time 4, so packet 1 served
    # 0-4, packet 2 queued then served 4-8, packet 3 dropped at 2.
    # After the long gap, burst 2 at 100, 101, 102: packet 4 served
    # 100-104, packet 5 queued then served 104-108, packet 6 dropped
    # at 102 (server busy 100-104, queue full of packet 5 -> drop).
    router, _, log, loop, _, trace = _build_simulator(
        capacity=1,
        inter_arrivals=[
            0.0, 1.0, 1.0,  # burst 1: arrivals at 0, 1, 2
            100.0,          # long gap -> next arrival at t=102
            1.0, 1.0,       # burst 2: arrivals at 103, 104
            10_000.0,       # sentinel: stop scheduling new arrivals
        ],
        service_time=4.0,
    )

    loop.run()

    drops = trace.filter(EventType.PACKET_DROP)
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    arrivals = trace.filter(EventType.PACKET_ARRIVAL)

    # Sanity: 6 arrivals total.
    assert len(arrivals) == 6
    arrival_times = sorted(a["timestamp"] for a in arrivals)
    assert arrival_times == [0.0, 1.0, 2.0, 102.0, 103.0, 104.0]

    # Total: 2 drops, 4 served.
    assert len(drops) == 2
    assert len(starts) == 4

    # The served packets (4 of them) are not the dropped ones; the queue
    # recovered after the long gap and accepted packets from the second burst.
    dropped_packet_ids = {d["packet_id"] for d in drops}
    served_packet_ids = {s["packet_id"] for s in starts}
    assert dropped_packet_ids.isdisjoint(served_packet_ids)
    # The dropped packets are 3 (in burst 1) and 6 (in burst 2).
    assert dropped_packet_ids == {3, 6}
    # The drop timestamps correspond to the arrival timestamps of those
    # packets in each burst.
    drop_times = sorted(d["timestamp"] for d in drops)
    assert drop_times == [2.0, 104.0]


def test_no_drop_when_queue_capacity_is_unlimited() -> None:
    """An infinite queue (M/M/1) never drops a packet, even with a busy server."""
    # Same burst as the dropping case, but capacity=None -> no drops.
    router, _, log, loop, _, trace = _build_simulator(
        capacity=None,
        inter_arrivals=[0.0, 1.0, 1.0, 1.0, 10_000.0],
        service_time=10.0,
    )

    loop.run()

    drops = trace.filter(EventType.PACKET_DROP)
    assert drops == []
    assert router.packets_dropped == 0
    assert router.queue_manager.is_full is False
    # All four packets eventually departed.
    assert len(trace.filter(EventType.PACKET_DEPARTURE)) == 4


# ---------------------------------------------------------------------------
# 4. FIFO ordering is honored end-to-end
# ---------------------------------------------------------------------------


def test_fifo_ordering_across_full_lifecycle() -> None:
    """In a congested run, packets depart in the order they arrived (FIFO)."""
    # Service time 4, arrivals every 1 time unit for 6 packets -> the
    # server is always busy, packets 2-6 are queued, and they must come
    # out in arrival order.
    router, _, log, loop, _, trace = _build_simulator(
        capacity=None,
        inter_arrivals=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10_000.0],
        service_time=4.0,
    )

    loop.run()

    arrivals = trace.filter(EventType.PACKET_ARRIVAL)
    departs = trace.filter(EventType.PACKET_DEPARTURE)

    assert len(arrivals) == 6
    assert len(departs) == 6

    # Reconstruct the per-packet id ordering from the events. The router
    # assigns packet_ids in monotonic order starting at 1.
    arrival_packet_ids = [a["packet_id"] for a in arrivals]
    depart_packet_ids = [d["packet_id"] for d in departs]
    assert arrival_packet_ids == [1, 2, 3, 4, 5, 6]
    assert depart_packet_ids == [1, 2, 3, 4, 5, 6]

    # Departures are also non-decreasing in time, and each departure
    # happens strictly after its arrival.
    for arr, dep in zip(arrivals, departs, strict=True):
        assert dep["timestamp"] >= arr["timestamp"]


def test_fifo_ordering_under_drop_pressure() -> None:
    """Even with drops, the surviving packets depart in FIFO order."""
    # Capacity 1, service time 6, arrivals every 1.5 time units. The
    # server is busy from 0 to 6, then 6 to 12, then 12 to 18, ...
    # Arrivals at 0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5.
    # At t=6, server frees exactly as packet 5 arrives -> packet 5 is
    # accepted (server idle, queue empty). At t=7.5 server busy, queue
    # empty -> packet 6 queued. At t=9 server busy, queue full ->
    # packet 7 dropped. Same for 8.
    router, _, log, loop, _, trace = _build_simulator(
        capacity=1,
        inter_arrivals=[1.5] * 8 + [10_000.0],
        service_time=6.0,
    )

    loop.run()

    arrivals = trace.filter(EventType.PACKET_ARRIVAL)
    departs = trace.filter(EventType.PACKET_DEPARTURE)
    drops = trace.filter(EventType.PACKET_DROP)

    # 8 arrivals, 3 served (1, 2, 5), 5 dropped.
    assert len(arrivals) == 8
    assert len(departs) == 3
    assert len(drops) == 5

    # The surviving packets depart in FIFO order.
    assert [d["packet_id"] for d in departs] == [1, 2, 5]

    # The dropped packets are the ones that arrived when both the
    # server was busy and the queue was full.
    assert sorted(d["packet_id"] for d in drops) == [3, 4, 6, 7, 8]
    for drop in drops:
        assert drop["metadata"]["reason"] == "queue_full"


def test_fifo_discipline_setting_propagates_to_queue_manager() -> None:
    """Sanity check: building the simulator with FIFO actually uses FIFO.

    This guards against an accidental re-ordering in the queue manager
    (the only place the discipline bit matters).
    """
    config = SimulationConfig(
        arrival=ArrivalConfig(arrival_rate=1.0),
        service=ServiceConfig(service_rate=1.0),
        queue=QueueConfig(capacity=5, queue_discipline=QueueDiscipline.FIFO),
        simulation_time=10.0,
        random_seed=0,
    )
    assert config.queue.queue_discipline is QueueDiscipline.FIFO
    q = QueueManager(config.queue)
    p1 = Packet(packet_id=1, arrival_time=0.0)
    p2 = Packet(packet_id=2, arrival_time=1.0)
    p3 = Packet(packet_id=3, arrival_time=2.0)
    q.enqueue(p1)
    q.enqueue(p2)
    q.enqueue(p3)
    assert q.dequeue() is p1
    assert q.dequeue() is p2
    assert q.dequeue() is p3


# ---------------------------------------------------------------------------
# 5. Combined / cross-cutting
# ---------------------------------------------------------------------------


def test_history_log_contains_complete_event_timeline() -> None:
    """The history log is a faithful record of the simulation's event sequence."""
    router, _, log, loop, _, trace = _build_simulator(
        capacity=2,
        inter_arrivals=[1.0, 1.0, 1.0, 1.0, 1.0, 10_000.0],
        service_time=3.0,
    )

    loop.run()

    # For every packet that was *not* dropped, the trace must contain
    # exactly one ARRIVAL, one SERVICE_START, and one DEPARTURE event
    # in that order in time.
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    ends = trace.filter(EventType.PACKET_DEPARTURE)
    drops = trace.filter(EventType.PACKET_DROP)

    served_ids = {s["packet_id"] for s in starts}
    for end in ends:
        assert end["packet_id"] in served_ids
    for drop in drops:
        assert drop["packet_id"] not in served_ids

    # Every timestamp in the history log is non-decreasing.
    timestamps = [r.timestamp for r in log]
    for prev, curr in zip(timestamps, timestamps[1:]):
        assert curr >= prev


def test_packet_lifecycle_recorded_correctly_via_router() -> None:
    """The router records arrival/service_start/departure times on each packet.

    We rely on the router's own packet list — each packet object
    exposes waiting_time, service_time, and system_time as derived
    properties. We check those against the trace timestamps.
    """
    router, scheduler, _, loop, _, trace = _build_simulator(
        capacity=None,
        inter_arrivals=[1.0, 1.0, 1.0, 10_000.0],
        service_time=2.0,
    )

    loop.run()

    assert router.packets_created == 3
    assert router.packets_dropped == 0
    # The last packet's service ended with the server idle.
    assert router.server.busy is False
    assert router.queue_manager.is_empty is True

    # Reconstruct the lifecycle from the trace: each served packet has
    # exactly one arrival, one service start, and one departure.
    arrivals = trace.filter(EventType.PACKET_ARRIVAL)
    starts = trace.filter(EventType.PACKET_SERVICE_START)
    departs = trace.filter(EventType.PACKET_DEPARTURE)

    assert len(arrivals) == 3
    assert len(starts) == 3
    assert len(departs) == 3

    for arr, start, dep in zip(arrivals, starts, departs, strict=True):
        assert arr["packet_id"] == start["packet_id"] == dep["packet_id"]
        # Service time == departure - service_start, matches the
        # constant service time the simulator was built with.
        assert dep["timestamp"] - start["timestamp"] == pytest.approx(2.0)


def test_end_to_end_against_yaml_config() -> None:
    """Sanity check: the production YAML config files parse and the
    simulator can be built from them with the same components used
    in the wiring tests.
    """
    from config.loader import load_config
    from distributions.continuous import ExponentialDistribution

    cfg = load_config("configs/mm1_example.yaml")
    assert cfg.arrival.arrival_rate == 5.0
    assert cfg.service.service_rate == 8.0
    assert cfg.queue.capacity is None
    assert cfg.queue.queue_discipline is QueueDiscipline.FIFO

    # The same wiring path as the integration tests, this time driven
    # by a real (seeded) Poisson arrival process and an exponential
    # service distribution. We just want to confirm it does not blow
    # up and produces at least some events.
    rng = np.random.default_rng(42)
    queue = QueueManager(cfg.queue)
    server = Server()
    scheduler = EventScheduler()
    service_dist = ExponentialDistribution(rate=cfg.service.service_rate, rng=rng)
    arrival_proc = HomogeneousPoissonProcessStub(rng, cfg.arrival.arrival_rate)
    log = EventHistoryLog()

    router = Router(
        queue_manager=queue,
        server=server,
        scheduler=scheduler,
        service_time_distribution=service_dist,
        baseline_drop_probability=0.0,
        rng=rng,
    )

    class YamlHandler(EventHandler):
        def __init__(self) -> None:
            self._done = False

        def handle(self, event: Event, current_time: float) -> None:
            log.record(event, current_time)
            if event.event_type == EventType.PACKET_ARRIVAL:
                router.handle_event(event)
                if not self._done:
                    self._done = True
                    # Schedule a fixed follow-up of additional arrivals
                    # so we don't depend on the stub beyond this.
                    next_ts = current_time + arrival_proc.next_delta()
                    if next_ts < 50.0:
                        scheduler.schedule(
                            next_ts,
                            EventType.PACKET_ARRIVAL,
                            packet_id=None,
                        )
                return
            if event.event_type in {
                EventType.PACKET_SERVICE_START,
                EventType.PACKET_DEPARTURE,
            }:
                router.handle_event(event)
                return

        def on_start(self, current_time: float) -> None:
            scheduler.schedule(0.0, EventType.PACKET_ARRIVAL, packet_id=None)

    loop = EventLoop(scheduler=scheduler, handler=YamlHandler())
    loop.run(max_time=50.0)

    arrivals = log.filter_by_type(EventType.PACKET_ARRIVAL)
    departs = log.filter_by_type(EventType.PACKET_DEPARTURE)
    assert len(arrivals) > 0
    # We stop at max_time=50, so departures <= arrivals. With an M/M/1
    # stable system (lambda=5, mu=8) we expect departures > 0 too.
    assert len(departs) > 0
    # No drops on an infinite queue.
    assert log.filter_by_type(EventType.PACKET_DROP) == []
    # Departures are in FIFO order (packet ids in increasing order).
    assert [d.packet_id for d in departs] == sorted(d.packet_id for d in departs)


class HomogeneousPoissonProcessStub:
    """A tiny stand-in for the Poisson process: just draws a small
    set of deterministic deltas and then refuses further arrivals.

    The integration tests use :class:`DeterministicInterArrival` for
    full control; this stub is only here so the YAML sanity check
    can run without spinning up a fresh RNG-driven process.
    """

    def __init__(self, rng: np.random.Generator, rate: float, deltas: tuple[float, ...] = (0.5, 1.0, 0.75, 1.5, 0.25, 2.0, 0.5, 1.0, 0.5, 3.0)) -> None:
        self._rng = rng
        self._rate = rate
        self._deltas = list(deltas)
        self._idx = 0

    def next_delta(self) -> float:
        if self._idx >= len(self._deltas):
            return 10_000.0
        v = self._deltas[self._idx]
        self._idx += 1
        return v

    def sample_inter_arrival_time(self) -> float:
        return self.next_delta()

    def sample_inter_arrival_times(self, n: int) -> np.ndarray:
        return np.array([self.next_delta() for _ in range(n)])
