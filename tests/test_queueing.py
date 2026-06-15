"""Tests for queue manager."""

import pytest

from config.models import QueueConfig, QueueDiscipline
from queueing.manager import QueueManager
from simulation.packet import Packet


def _make_packet(packet_id: int) -> Packet:
    """Create a test packet with a given ID."""
    return Packet(packet_id=packet_id, arrival_time=0.0)


class TestQueueManagerInfinite:
    """Tests for M/M/1 infinite-capacity queue."""

    def test_enqueue_dequeue_fifo(self) -> None:
        """FIFO discipline preserves arrival order."""
        config = QueueConfig(capacity=None, queue_discipline=QueueDiscipline.FIFO)
        queue = QueueManager(config)

        p1 = _make_packet(1)
        p2 = _make_packet(2)
        assert queue.enqueue(p1) is True
        assert queue.enqueue(p2) is True
        assert queue.size == 2

        assert queue.dequeue().packet_id == 1
        assert queue.dequeue().packet_id == 2
        assert queue.is_empty

    def test_never_full(self) -> None:
        """Infinite queue is never full."""
        config = QueueConfig(capacity=None)
        queue = QueueManager(config)
        for i in range(100):
            queue.enqueue(_make_packet(i))
        assert queue.is_full is False

    def test_dequeue_empty_raises(self) -> None:
        """Dequeuing from empty queue raises IndexError."""
        queue = QueueManager(QueueConfig(capacity=None))
        with pytest.raises(IndexError):
            queue.dequeue()


class TestQueueManagerFinite:
    """Tests for M/M/1/K finite-capacity queue."""

    def test_capacity_constraint(self) -> None:
        """Queue rejects packets when at capacity."""
        config = QueueConfig(capacity=2)
        queue = QueueManager(config)

        assert queue.enqueue(_make_packet(1)) is True
        assert queue.enqueue(_make_packet(2)) is True
        assert queue.enqueue(_make_packet(3)) is False
        assert queue.size == 2
        assert queue.total_dropped == 1

    def test_drop_count_increments(self) -> None:
        """Each rejected packet increments drop counter."""
        config = QueueConfig(capacity=1)
        queue = QueueManager(config)
        queue.enqueue(_make_packet(1))
        queue.enqueue(_make_packet(2))
        queue.enqueue(_make_packet(3))
        assert queue.total_dropped == 2
        assert queue.total_enqueued == 1

    def test_peek_returns_head(self) -> None:
        """peek returns front packet without removal."""
        config = QueueConfig(capacity=3)
        queue = QueueManager(config)
        p = _make_packet(10)
        queue.enqueue(p)
        assert queue.peek() is p
        assert queue.size == 1

    def test_clear_resets_state(self) -> None:
        """clear empties queue and resets counters."""
        config = QueueConfig(capacity=2)
        queue = QueueManager(config)
        queue.enqueue(_make_packet(1))
        queue.clear()
        assert queue.is_empty
        assert queue.total_enqueued == 0
        assert queue.total_dropped == 0


class TestPacketModel:
    """Tests for Packet domain model."""

    def test_waiting_time(self) -> None:
        """Waiting time computed correctly."""
        packet = Packet(packet_id=1, arrival_time=1.0)
        packet.mark_service_start(4.0)
        assert packet.waiting_time == 3.0

    def test_service_time(self) -> None:
        """Service time computed correctly."""
        packet = Packet(packet_id=1, arrival_time=1.0)
        packet.mark_service_start(2.0)
        packet.mark_departure(5.0)
        assert packet.service_time == 3.0

    def test_system_time(self) -> None:
        """System time computed correctly."""
        packet = Packet(packet_id=1, arrival_time=1.0)
        packet.mark_service_start(3.0)
        packet.mark_departure(7.0)
        assert packet.system_time == 6.0

    def test_mark_dropped(self) -> None:
        """Drop metadata recorded correctly."""
        packet = Packet(packet_id=1, arrival_time=0.0)
        packet.mark_dropped(drop_time=2.5, reason="queue_full")
        assert packet.dropped is True
        assert packet.drop_time == 2.5
        assert packet.drop_reason == "queue_full"
