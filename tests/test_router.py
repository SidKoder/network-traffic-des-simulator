"""Tests for the router controller arrival and service flow."""

import numpy as np

from config.models import QueueConfig
from events.scheduler import EventScheduler
from events.types import EventType
from queueing.manager import QueueManager
from simulation.router import Router
from simulation.server import Server


class FixedServiceTime:
    """Deterministic service-time distribution for router tests."""

    def __init__(self, service_time: float) -> None:
        self.service_time = service_time

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(size, self.service_time)


def _router(
    *,
    capacity: int | None = 2,
    service_time: float = 5.0,
    baseline_drop_probability: float = 0.0,
) -> Router:
    return Router(
        queue_manager=QueueManager(QueueConfig(capacity=capacity)),
        server=Server(),
        scheduler=EventScheduler(),
        service_time_distribution=FixedServiceTime(service_time),
        baseline_drop_probability=baseline_drop_probability,
    )


def test_arrival_schedules_service_start_when_server_is_idle() -> None:
    """An accepted packet schedules immediate service when the CPU is idle."""
    router = _router(service_time=2.5)

    packet = router.handle_arrival(current_time=10.0)

    assert packet.packet_id == 1
    assert packet.dropped is False
    assert router.queue_manager.is_empty is True
    assert router.server.busy is False
    assert packet.service_start_time is None

    service_start = router.scheduler.next_event()
    assert service_start.event_type == EventType.PACKET_SERVICE_START
    assert service_start.timestamp == 10.0
    assert service_start.packet_id == packet.packet_id

    started = router.handle_event(service_start)
    departure = router.scheduler.next_event()

    assert started is packet
    assert router.server.busy is True
    assert router.server.current_packet is packet
    assert router.server.busy_start_time == 10.0
    assert packet.service_start_time == 10.0
    assert departure.event_type == EventType.PACKET_DEPARTURE
    assert departure.timestamp == 12.5
    assert departure.packet_id == packet.packet_id
    assert departure.metadata["service_time"] == 2.5


def test_arrival_waits_in_queue_when_server_is_busy() -> None:
    """A packet accepted while CPU is busy remains buffered in router memory."""
    router = _router(capacity=2)
    first = router.handle_arrival(current_time=0.0)
    router.handle_event(router.scheduler.next_event())

    second = router.handle_arrival(current_time=1.0)

    assert router.server.busy is True
    assert router.server.current_packet is first
    assert router.queue_manager.size == 1
    assert router.queue_manager.peek() is second
    assert second.service_start_time is None
    assert router.scheduler.pending_count == 1


def test_baseline_drop_happens_before_queue_capacity_check() -> None:
    """Baseline drop rejects the packet without touching queue or server."""
    router = _router(capacity=1, baseline_drop_probability=1.0)

    packet = router.handle_arrival(current_time=3.0)

    assert packet.dropped is True
    assert packet.drop_time == 3.0
    assert packet.drop_reason == "baseline_drop"
    assert router.queue_manager.is_empty is True
    assert router.queue_manager.total_dropped == 0
    assert router.server.busy is False
    assert router.packets_created == 1
    assert router.packets_dropped == 1

    drop_event = router.scheduler.next_event()
    assert drop_event.event_type == EventType.PACKET_DROP
    assert drop_event.timestamp == 3.0
    assert drop_event.packet_id == packet.packet_id
    assert drop_event.metadata["reason"] == "baseline_drop"


def test_queue_full_drop_happens_when_router_memory_is_full() -> None:
    """A full finite queue drops the next non-baseline-dropped arrival."""
    router = _router(capacity=1, service_time=10.0)

    in_service = router.handle_arrival(current_time=0.0)
    router.handle_event(router.scheduler.next_event())
    waiting = router.handle_arrival(current_time=0.1)
    dropped = router.handle_arrival(current_time=0.2)

    assert router.server.current_packet is in_service
    assert router.queue_manager.size == 1
    assert router.queue_manager.peek() is waiting
    assert dropped.dropped is True
    assert dropped.drop_reason == "queue_full"
    assert dropped.drop_time == 0.2
    assert router.queue_manager.total_dropped == 1
    assert router.packets_dropped == 1

    events = [router.scheduler.next_event() for _ in range(router.scheduler.pending_count)]
    drop_events = [event for event in events if event.event_type == EventType.PACKET_DROP]
    assert len(drop_events) == 1
    assert drop_events[0].packet_id == dropped.packet_id
    assert drop_events[0].metadata["reason"] == "queue_full"


def test_departure_finishes_current_packet_and_starts_next_waiting_packet() -> None:
    """After service completion, the router immediately serves the next packet."""
    router = _router(capacity=1, service_time=5.0)
    first = router.handle_arrival(current_time=0.0)
    router.handle_event(router.scheduler.next_event())
    second = router.handle_arrival(current_time=1.0)

    assert router.queue_manager.size == 1
    assert router.queue_manager.is_full is True

    departure = router.scheduler.next_event()
    assert departure.event_type == EventType.PACKET_DEPARTURE

    completed = router.handle_event(departure)

    assert completed is first
    assert first.departure_time == 5.0
    assert router.server.busy is False
    assert router.server.current_packet is None
    assert router.server.busy_start_time is None
    assert router.queue_manager.is_empty is True
    assert router.queue_manager.size == 0
    assert router.queue_manager.is_full is False

    next_service_start = router.scheduler.next_event()
    assert next_service_start.event_type == EventType.PACKET_SERVICE_START
    assert next_service_start.packet_id == second.packet_id
    assert next_service_start.timestamp == 5.0

    router.handle_event(next_service_start)

    next_departure = router.scheduler.next_event()
    assert router.server.busy is True
    assert router.server.current_packet is second
    assert router.server.busy_start_time == 5.0
    assert second.service_start_time == 5.0
    assert next_service_start.event_type == EventType.PACKET_SERVICE_START
    assert next_service_start.packet_id == second.packet_id
    assert next_service_start.timestamp == 5.0
    assert next_departure.event_type == EventType.PACKET_DEPARTURE
    assert next_departure.packet_id == second.packet_id
    assert next_departure.timestamp == 10.0


def test_departure_with_empty_queue_records_departure_and_stops() -> None:
    """If no packet is waiting, departure only frees the server."""
    router = _router(capacity=1, service_time=4.0)
    packet = router.handle_arrival(current_time=2.0)
    service_start = router.scheduler.next_event()
    router.handle_event(service_start)

    departure = router.scheduler.next_event()
    assert departure.event_type == EventType.PACKET_DEPARTURE

    completed = router.handle_event(departure)

    assert completed is packet
    assert packet.departure_time == 6.0
    assert router.server.busy is False
    assert router.server.current_packet is None
    assert router.server.busy_start_time is None
    assert router.queue_manager.is_empty is True
    assert router.queue_manager.size == 0
    assert router.queue_manager.is_full is False
    assert router.scheduler.is_empty() is True
