"""Priority-queue event scheduler."""

import heapq
from typing import Iterator

from events.types import Event, EventType


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
        payload: dict | None = None,
    ) -> Event:
        """Insert an event into the priority queue.

        Parameters:
            timestamp: Simulation time for the event.
            event_type: Type of event to schedule.
            payload: Optional event-associated data.

        Returns:
            The scheduled Event instance.
        """
        event = Event(
            timestamp=timestamp,
            sequence=self._sequence_counter,
            event_type=event_type,
            payload=payload or {},
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
