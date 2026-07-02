"""Priority-queue event scheduler."""

import heapq
from typing import Any
from typing import Iterator

from events.event import Event
from events.types import EventType


class EventScheduler:
    """Manages a priority queue of future simulation events.

    Events are processed in ascending timestamp order. Equal timestamps
    are resolved by insertion sequence (FIFO among ties).
    """

    def __init__(self) -> None:
        """Initialize an empty event scheduler."""
        self._queue: list[Event] = []
        self._sequence_counter: int = 0

    @property
    def pending_count(self) -> int:
        """Return the number of events waiting to be processed.

        Returns:
            Count of scheduled events.
        """
        return len(self._queue)

    def schedule(
        self,
        timestamp: float,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        *,
        packet_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> Event:
        """Insert an event into the priority queue.

        Parameters:
            timestamp: Simulation time for the event.
            event_type: Type of event to schedule.
            payload: Legacy name for event metadata.
            packet_id: Identifier of the associated packet, if any.
            metadata: Optional event-associated data.
            event_id: Optional caller-provided unique event identifier.

        Returns:
            The scheduled Event instance.
        """
        if payload is not None and metadata is not None:
            raise ValueError("Use either payload or metadata, not both")

        event_kwargs: dict[str, Any] = {
            "timestamp": timestamp,
            "sequence": self._sequence_counter,
            "event_type": event_type,
            "packet_id": packet_id,
            "metadata": metadata if metadata is not None else (payload or {}),
        }
        if event_id is not None:
            event_kwargs["event_id"] = event_id

        event = Event(
            **event_kwargs,
        )
        heapq.heappush(self._queue, event)
        self._sequence_counter += 1
        return event

    def next_event(self) -> Event:
        """Remove and return the earliest pending event.

        Returns:
            The next event in chronological order.

        Raises:
            IndexError: If no events are scheduled.
        """
        if not self._queue:
            raise IndexError("No events scheduled")
        return heapq.heappop(self._queue)

    def peek(self) -> Event | None:
        """Inspect the earliest pending event without removing it.

        Returns:
            The next event, or None if the queue is empty.
        """
        return self._queue[0] if self._queue else None

    def clear(self) -> None:
        """Remove all scheduled events."""
        self._queue.clear()
        self._sequence_counter = 0

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events in chronological order (destructive).

        Yields:
            Events in ascending timestamp order.
        """
        while self._queue:
            yield heapq.heappop(self._queue)

    def __len__(self) -> int:
        """Return the number of pending events.

        Returns:
            Event count.
        """
        return len(self._queue)
