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
