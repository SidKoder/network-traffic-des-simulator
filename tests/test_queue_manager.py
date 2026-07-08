"""Focused tests for the router-memory queue manager."""

import pytest

from config.models import QueueConfig, QueueDiscipline
from queueing.manager import QueueManager
from simulation.packet import Packet


def _packet(packet_id: int) -> Packet:
    """Create a packet whose ID makes queue ordering easy to assert."""
    return Packet(packet_id=packet_id, arrival_time=float(packet_id))


def _queued_packet_ids(queue: QueueManager) -> list[int]:
    """Drain the queue and return packet IDs in service order."""
    packet_ids: list[int] = []
    while not queue.is_empty:
        packet_ids.append(queue.dequeue().packet_id)
    return packet_ids


def test_enqueue_accepts_packet_and_updates_queue_state() -> None:
    """enqueue stores an accepted packet and updates size/enqueue counters."""
    queue = QueueManager(QueueConfig(capacity=3))
    packet = _packet(1)

    accepted = queue.enqueue(packet)

    assert accepted is True
    assert queue.size == 1
    assert queue.is_empty is False
    assert queue.is_full is False
    assert queue.total_enqueued == 1
    assert queue.total_dropped == 0
    assert queue.peek() is packet


def test_dequeue_returns_next_packet_and_removes_it() -> None:
    """dequeue returns the buffered packet and leaves the queue empty."""
    queue = QueueManager(QueueConfig(capacity=1))
    packet = _packet(1)
    queue.enqueue(packet)

    dequeued = queue.dequeue()

    assert dequeued is packet
    assert queue.size == 0
    assert queue.is_empty is True
    assert queue.peek() is None


def test_dequeue_empty_queue_raises_index_error() -> None:
    """dequeue fails clearly when no packet is buffered."""
    queue = QueueManager(QueueConfig(capacity=1))

    with pytest.raises(IndexError, match="Cannot dequeue from empty queue"):
        queue.dequeue()


def test_finite_queue_reports_full_only_when_capacity_is_reached() -> None:
    """finite router memory becomes full exactly at configured capacity."""
    queue = QueueManager(QueueConfig(capacity=2))

    assert queue.is_full is False

    assert queue.enqueue(_packet(1)) is True
    assert queue.is_full is False

    assert queue.enqueue(_packet(2)) is True
    assert queue.size == 2
    assert queue.is_full is True


def test_enqueue_rejects_packet_when_queue_is_full() -> None:
    """a full finite queue rejects the next packet without changing contents."""
    queue = QueueManager(QueueConfig(capacity=2))
    first = _packet(1)
    second = _packet(2)
    rejected = _packet(3)

    assert queue.enqueue(first) is True
    assert queue.enqueue(second) is True

    accepted = queue.enqueue(rejected)

    assert accepted is False
    assert queue.is_full is True
    assert queue.size == 2
    assert queue.total_enqueued == 2
    assert queue.total_dropped == 1
    assert _queued_packet_ids(queue) == [1, 2]


def test_infinite_queue_never_reports_full() -> None:
    """infinite M/M/1 router memory accepts packets without a full state."""
    queue = QueueManager(QueueConfig(capacity=None))

    for packet_id in range(1, 11):
        assert queue.enqueue(_packet(packet_id)) is True

    assert queue.size == 10
    assert queue.is_full is False
    assert queue.total_enqueued == 10
    assert queue.total_dropped == 0


def test_fifo_order_is_maintained_across_enqueue_and_dequeue() -> None:
    """FIFO queue discipline preserves arrival order through service."""
    queue = QueueManager(
        QueueConfig(capacity=5, queue_discipline=QueueDiscipline.FIFO)
    )

    for packet_id in [101, 102, 103, 104, 105]:
        assert queue.enqueue(_packet(packet_id)) is True

    assert _queued_packet_ids(queue) == [101, 102, 103, 104, 105]
    assert queue.is_empty is True

