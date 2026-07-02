"""Tests for event types and scheduler."""

import pytest

from events.event import Event
from events.scheduler import EventScheduler
from events.types import EventType


class TestEventScheduler:
    """Tests for EventScheduler priority queue behavior."""

    def test_empty_scheduler_raises(self) -> None:
        """next_event on empty scheduler raises IndexError."""
        scheduler = EventScheduler()
        with pytest.raises(IndexError):
            scheduler.next_event()

    def test_schedule_increases_count(self) -> None:
        """Scheduling events increases pending count."""
        scheduler = EventScheduler()
        scheduler.schedule(1.0, EventType.PACKET_ARRIVAL)
        assert scheduler.pending_count == 1

    def test_events_processed_in_timestamp_order(self) -> None:
        """Events are dequeued in ascending timestamp order."""
        scheduler = EventScheduler()
        scheduler.schedule(5.0, EventType.PACKET_DEPARTURE, {"id": 3})
        scheduler.schedule(1.0, EventType.PACKET_ARRIVAL, {"id": 1})
        scheduler.schedule(3.0, EventType.PACKET_SERVICE_START, {"id": 2})

        timestamps = [scheduler.next_event().timestamp for _ in range(3)]
        assert timestamps == [1.0, 3.0, 5.0]

    def test_tie_breaking_by_sequence(self) -> None:
        """Equal timestamps are resolved by insertion order."""
        scheduler = EventScheduler()
        first = scheduler.schedule(2.0, EventType.PACKET_ARRIVAL, {"id": 1})
        second = scheduler.schedule(2.0, EventType.PACKET_ARRIVAL, {"id": 2})

        popped_first = scheduler.next_event()
        popped_second = scheduler.next_event()

        assert popped_first.sequence < popped_second.sequence
        assert popped_first.payload["id"] == 1
        assert popped_second.payload["id"] == 2
        assert first.sequence == 0
        assert second.sequence == 1

    def test_peek_does_not_remove(self) -> None:
        """peek returns earliest event without dequeuing."""
        scheduler = EventScheduler()
        scheduler.schedule(4.0, EventType.PACKET_DROP)
        peeked = scheduler.peek()
        assert peeked is not None
        assert peeked.timestamp == 4.0
        assert scheduler.pending_count == 1

    def test_clear_removes_all(self) -> None:
        """clear empties the scheduler."""
        scheduler = EventScheduler()
        scheduler.schedule(1.0, EventType.PACKET_ARRIVAL)
        scheduler.schedule(2.0, EventType.PACKET_DEPARTURE)
        scheduler.clear()
        assert scheduler.pending_count == 0
        assert scheduler.peek() is None

    def test_iteration_drains_queue(self) -> None:
        """Iterating over scheduler yields all events in order."""
        scheduler = EventScheduler()
        scheduler.schedule(3.0, EventType.PACKET_DEPARTURE)
        scheduler.schedule(1.0, EventType.PACKET_ARRIVAL)
        events = list(scheduler)
        assert [e.timestamp for e in events] == [1.0, 3.0]
        assert len(scheduler) == 0

    def test_all_event_types_supported(self) -> None:
        """All four event types can be scheduled."""
        scheduler = EventScheduler()
        for event_type in EventType:
            scheduler.schedule(0.0, event_type)
        assert scheduler.pending_count == 4

    def test_requested_timestamp_input(self) -> None:
        """The requested timestamps leave the queue in chronological order."""
        scheduler = EventScheduler()
        timestamps = [1.2, 0.3, 2.8, 2.1]

        for packet_id, timestamp in enumerate(timestamps):
            scheduler.add_event(
                Event(
                    timestamp=timestamp,
                    event_type=EventType.PACKET_ARRIVAL,
                    packet_id=packet_id,
                )
            )

        assert scheduler.queue_size() == 4
        assert scheduler.view_next_event() is not None
        assert scheduler.view_next_event().timestamp == 0.3
        assert [
            scheduler.remove_event().timestamp for _ in range(len(timestamps))
        ] == [0.3, 1.2, 2.1, 2.8]
        assert scheduler.is_empty()

    def test_duplicate_timestamps_are_not_lost(self) -> None:
        """Distinct events at the same timestamp remain in the queue."""
        scheduler = EventScheduler()
        scheduler.schedule(1.2, EventType.PACKET_ARRIVAL, packet_id=1)
        scheduler.schedule(1.2, EventType.PACKET_ARRIVAL, packet_id=2)

        assert scheduler.queue_size() == 2
        assert [scheduler.remove_event().packet_id for _ in range(2)] == [1, 2]

    def test_explicit_operations_handle_empty_queue(self) -> None:
        """Empty queues report their state and reject removal cleanly."""
        scheduler = EventScheduler()

        assert scheduler.is_empty()
        assert scheduler.queue_size() == 0
        assert scheduler.view_next_event() is None
        with pytest.raises(IndexError, match="No events scheduled"):
            scheduler.remove_event()


class TestEvent:
    """Tests for the core simulation event model."""

    def test_contains_action_context(self) -> None:
        """An event records identity, packet, type, and metadata."""
        event = Event(
            timestamp=1.5,
            event_type=EventType.PACKET_ARRIVAL,
            packet_id=42,
            metadata={"source": "generator"},
        )

        assert event.event_id
        assert event.packet_id == 42
        assert event.metadata == {"source": "generator"}

    def test_compares_by_timestamp(self) -> None:
        """Earlier timestamps compare before later timestamps."""
        earlier = Event(1.0, EventType.PACKET_ARRIVAL)
        later = Event(2.0, EventType.PACKET_ARRIVAL)

        assert earlier < later
        assert later > earlier

    def test_sequence_breaks_timestamp_ties(self) -> None:
        """Sequence is the only tie-breaker for equal timestamps."""
        first = Event(
            1.0,
            EventType.PACKET_DEPARTURE,
            packet_id=99,
            sequence=3,
        )
        second = Event(
            1.0,
            EventType.PACKET_ARRIVAL,
            packet_id=1,
            sequence=4,
        )

        assert first < second

    def test_scheduler_populates_new_event_fields(self) -> None:
        """The scheduler propagates packet and metadata context."""
        scheduler = EventScheduler()
        event = scheduler.schedule(
            1.0,
            EventType.PACKET_ARRIVAL,
            packet_id=7,
            metadata={"queue": "primary"},
        )

        assert event.packet_id == 7
        assert event.metadata == {"queue": "primary"}
        assert event.payload is event.metadata
