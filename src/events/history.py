"""Event history log for debugging and post-mortem analysis.

The :class:`EventHistoryLog` records every event the engine has
processed — the simulation time at which it fired, the event type, the
associated packet id, and the originating event id. It is a passive
observer: it never mutates the event loop, never schedules follow-ups,
and never raises on duplicate or out-of-order timestamps (the
scheduler's tie-breaker keeps arrival order deterministic, but the log
makes that order explicit so it survives a crash).

The log is intentionally append-only and supports random access by
index, slicing, and JSON export for offline inspection.

Typical wiring with the engine::

    from events.history import EventHistoryLog
    from simulation.engine import EventLoop

    log = EventHistoryLog()

    class HistoryHandler:
        def handle(self, event, current_time):
            log.record(event, current_time)

    loop = EventLoop(handler=HistoryHandler())
    loop.run()
    print(log.to_json())

For richer observability, pass the same log to multiple handlers (a
printer, a metrics collector, etc.) and inspect the merged trail after
the run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from events.event import Event
from events.types import EventType


@dataclass(frozen=True)
class HistoryRecord:
    """A single entry in the :class:`EventHistoryLog`.

    The record is frozen so a log entry can never be mutated after
    capture — preserving the audit trail that the log is meant to
    provide.

    Attributes:
        timestamp: Simulation time at which the event was processed.
        event_type: Category of action recorded.
        packet_id: Identifier of the packet associated with the event,
            or ``None`` for events that are not packet-scoped.
        event_id: Globally unique identifier of the originating event.
    """

    timestamp: float
    event_type: EventType
    packet_id: int | None
    event_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record.

        Returns:
            A dict with ``event_type`` rendered as its string value so
            the result can be passed straight to :func:`json.dumps`.
        """
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return data


class EventHistoryLog:
    """Append-only log of processed simulation events.

    The log is meant to be wired into the engine as an
    :class:`~simulation.engine.EventHandler`. It records every event
    the engine dispatches and exposes simple inspection helpers
    (``__len__``, ``__iter__``, ``__getitem__``, ``records``) so callers
    can slice, iterate, or count entries without copying.

    The log has no built-in size cap; callers that run long
    simulations should periodically clear it via :meth:`clear` to bound
    memory use.
    """

    def __init__(self) -> None:
        """Initialize an empty event history log."""
        self._records: list[HistoryRecord] = []

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self._records)

    def __iter__(self) -> Iterator[HistoryRecord]:
        """Iterate over recorded events in the order they were processed."""
        return iter(self._records)

    def __getitem__(self, index: int | slice) -> HistoryRecord | list[HistoryRecord]:
        """Return a record by index or a list of records by slice.

        Parameters:
            index: Integer position or slice into the log.

        Returns:
            The :class:`HistoryRecord` at that position, or a list of
            records for a slice.
        """
        return self._records[index]

    @property
    def records(self) -> list[HistoryRecord]:
        """Return a copy of the recorded entries as a list.

        Returns:
            A list of :class:`HistoryRecord` in processing order.
        """
        return list(self._records)

    def record(self, event: Event, current_time: float) -> None:
        """Append ``event`` to the history log.

        Parameters:
            event: The event just processed by the engine.
            current_time: The simulation time at which it fired.
        """
        self._records.append(
            HistoryRecord(
                timestamp=current_time,
                event_type=event.event_type,
                packet_id=event.packet_id,
                event_id=event.event_id,
            )
        )

    def clear(self) -> None:
        """Remove all recorded entries from the log."""
        self._records.clear()

    def filter_by_packet(self, packet_id: int) -> list[HistoryRecord]:
        """Return every record associated with ``packet_id``.

        Parameters:
            packet_id: Packet identifier to filter on.

        Returns:
            Records, in processing order, whose ``packet_id`` matches.
        """
        return [record for record in self._records if record.packet_id == packet_id]

    def filter_by_type(self, event_type: EventType) -> list[HistoryRecord]:
        """Return every record whose ``event_type`` matches.

        Parameters:
            event_type: Event category to filter on.

        Returns:
            Records, in processing order, whose ``event_type`` matches.
        """
        return [record for record in self._records if record.event_type is event_type]

    def to_json(self) -> str:
        """Serialize the log to a JSON string.

        Returns:
            A JSON array of records, each rendered via
            :meth:`HistoryRecord.to_dict`.
        """
        return json.dumps([record.to_dict() for record in self._records])

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return the log as a list of plain dicts.

        Returns:
            A list of :class:`HistoryRecord` data, with ``event_type``
            rendered as its string value.
        """
        return [record.to_dict() for record in self._records]
