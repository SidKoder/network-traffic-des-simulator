"""Generic discrete-event simulation engine.

The engine coordinates three collaborators:

* :class:`simulation.clock.SimulationClock` — owns the current simulation
  time and rejects backward movement.
* :class:`events.scheduler.EventScheduler` — priority-queue of pending
  events ordered by ``(timestamp, sequence)``.
* :class:`EventHandler` — pluggable strategy invoked once per event. The
  default implementation just logs the event to standard output, but
  callers can subclass :class:`EventHandler` to mutate state, schedule
  follow-up events, or collect metrics.

The engine itself is intentionally generic: it does not know what an
event *means*, it only advances time and dispatches events to the
handler. Domain logic belongs in the handler, not in the engine.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterator

from events.event import Event
from events.scheduler import EventScheduler
from simulation.clock import SimulationClock

logger = logging.getLogger(__name__)


class EventHandler(ABC):
    """Interface for processing a single event.

    Implementations receive the current simulation time and the event
    being processed, and may return follow-up events to be enqueued in
    the scheduler. The default :class:`PrintEventHandler` simply logs the
    event to standard output — useful as a smoke test and as a
    reference for writing concrete handlers.

    The optional :meth:`on_start` and :meth:`on_stop` lifecycle hooks
    default to no-ops, so minimal handlers only need to implement
    :meth:`handle`. The engine invokes these hooks via ``getattr``, so
    duck-typed handlers (plain classes that expose ``handle``) work
    just as well as subclasses of this ABC.
    """

    @abstractmethod
    def handle(self, event: Event, current_time: float) -> Iterator[Event] | None:
        """Process ``event`` at ``current_time``.

        Parameters:
            event: The event just popped from the scheduler.
            current_time: The simulation time at which it fires.

        Returns:
            An optional iterable of follow-up events to schedule, or
            ``None`` if the handler has nothing to enqueue.
        """

    def on_start(self, current_time: float) -> None:
        """Hook called once before the loop begins processing events."""

    def on_stop(self, current_time: float) -> None:
        """Hook called once after the loop drains the scheduler."""


class PrintEventHandler(EventHandler):
    """Default handler that prints/logs each event as it is processed.

    The handler is useful for debugging and for smoke-testing the engine
    without any domain logic wired in. Output is emitted through the
    standard :mod:`logging` facility at ``INFO`` level; ``print`` is
    used as a fallback so the handler works even when logging is not
    configured.
    """

    def __init__(self, *, use_logging: bool = True) -> None:
        """Initialize the printer.

        Parameters:
            use_logging: When ``True`` (default), emit via the module
                logger at ``INFO`` level. When ``False``, write directly
                to standard output via :func:`print`.
        """
        self._use_logging = use_logging

    def handle(self, event: Event, current_time: float) -> Iterator[Event] | None:
        """Log ``event`` to standard output.

        Parameters:
            event: The event being processed.
            current_time: The simulation time at which it fires.

        Returns:
            ``None`` — the printer never schedules follow-up events.
        """
        message = (
            f"[t={current_time:.6f}] "
            f"event_id={event.event_id} "
            f"type={event.event_type.name} "
            f"packet_id={event.packet_id} "
            f"metadata={event.metadata}"
        )
        if self._use_logging:
            logger.info(message)
        else:
            print(message)
        return None


class EventLoop:
    """Generic event loop that advances time and dispatches events.

    The loop owns a clock and a scheduler; on each iteration it pops
    the earliest event, advances the clock to its timestamp, and hands
    the event to a handler. Follow-up events returned by the handler
    are re-injected into the scheduler, so the engine naturally
    supports event chains (departure → service start → arrival) without
    any domain knowledge.

    The loop terminates when the scheduler is empty or when the
    optional ``max_events`` / ``max_time`` limits are reached.
    """

    def __init__(
        self,
        scheduler: EventScheduler | None = None,
        clock: SimulationClock | None = None,
        handler: EventHandler | None = None,
    ) -> None:
        """Initialize the event loop.

        Parameters:
            scheduler: Priority queue of pending events. A new empty
                scheduler is created if omitted.
            clock: Simulation clock. A new clock starting at ``0.0`` is
                created if omitted.
            handler: Event handler. Defaults to
                :class:`PrintEventHandler` so the loop is usable out of
                the box.
        """
        self._scheduler = scheduler if scheduler is not None else EventScheduler()
        self._clock = clock if clock is not None else SimulationClock()
        self._handler: EventHandler = (
            handler if handler is not None else PrintEventHandler()
        )
        self._processed: int = 0

    @property
    def clock(self) -> SimulationClock:
        """Return the simulation clock used by the loop."""
        return self._clock

    @property
    def scheduler(self) -> EventScheduler:
        """Return the event scheduler used by the loop."""
        return self._scheduler

    @property
    def handler(self) -> EventHandler:
        """Return the currently configured event handler."""
        return self._handler

    @handler.setter
    def handler(self, handler: EventHandler) -> None:
        """Replace the event handler.

        Parameters:
            handler: New handler to dispatch events to.
        """
        self._handler = handler

    @property
    def processed_count(self) -> int:
        """Return the number of events processed so far in this loop."""
        return self._processed

    def add_event(self, event: Event) -> Event:
        """Schedule a follow-up event in the loop's scheduler.

        Parameters:
            event: Event to enqueue.

        Returns:
            The enqueued event.
        """
        return self._scheduler.add_event(event)

    def run(
        self,
        *,
        max_events: int | None = None,
        max_time: float | None = None,
    ) -> int:
        """Drive the event loop until the scheduler is empty.

        Parameters:
            max_events: Optional cap on the number of events to process
                before stopping early. ``None`` (default) processes
                every scheduled event.
            max_time: Optional cap on the simulation time the clock may
                reach. If the next event would fire after ``max_time``,
                the loop stops without processing it. ``None`` (default)
                imposes no time cap.

        Returns:
            The number of events processed during the run.

        Raises:
            ValueError: If ``max_events`` or ``max_time`` is negative.
        """
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be non-negative")
        if max_time is not None and max_time < 0:
            raise ValueError("max_time must be non-negative")

        on_start = getattr(self._handler, "on_start", None)
        on_stop = getattr(self._handler, "on_stop", None)
        if on_start is not None:
            on_start(self._clock.current_time)

        while not self._scheduler.is_empty():
            if max_events is not None and self._processed >= max_events:
                break

            next_event = self._scheduler.peek()
            if next_event is None:
                break
            if max_time is not None and next_event.timestamp > max_time:
                break

            event = self._scheduler.next_event()
            self._clock.advance_to(event.timestamp)

            follow_ups = self._handler.handle(event, self._clock.current_time)
            if follow_ups is not None:
                for follow_up in follow_ups:
                    self._scheduler.add_event(follow_up)

            self._processed += 1

        if on_stop is not None:
            on_stop(self._clock.current_time)
        return self._processed
