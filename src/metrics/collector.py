"""Passive event recorder for the DES engine.

The :class:`MetricsCollector` is a single-responsibility observer: it
records everything that happens during a simulation run and stores the
history. It deliberately performs **no computation** — it does not
compute averages, drop rates, throughput, utilization, or any other
derived metric. Its public surface is storage (append, iterate, slice,
filter) and serialization (to JSON / to dicts). Derived metrics belong
in a separate analytics layer that reads the collector's history.

The collector is intentionally **not** an
:class:`~simulation.engine.EventHandler` subclass. It is a pure storage
object; the caller wires it into the engine by writing a small handler
that dispatches each :class:`~events.types.EventType` to the matching
``record_*`` method. Typical wiring::

    from events.types import EventType
    from metrics import MetricsCollector
    from simulation.engine import EventHandler


    collector = MetricsCollector()


    class CollectingHandler(EventHandler):
        def __init__(self, router, collector):
            self._router = router
            self._collector = collector

        def handle(self, event, current_time):
            if event.event_type == EventType.PACKET_ARRIVAL:
                # The router assigns the packet id during handle_event,
                # so the router must be called *before* record_arrival.
                packet = self._router.handle_event(event)
                self._collector.record_arrival(
                    current_time, packet.packet_id
                )
                return None

            if event.event_type == EventType.PACKET_SERVICE_START:
                packet = self._router.handle_event(event)
                self._collector.record_service_start(
                    current_time, packet.packet_id, details=event.metadata
                )
                return None

            if event.event_type == EventType.PACKET_DEPARTURE:
                packet = self._router.handle_event(event)
                self._collector.record_departure(
                    current_time,
                    packet.packet_id,
                    details={"service_time": event.metadata.get("service_time")},
                )
                return None

            if event.event_type == EventType.PACKET_DROP:
                self._router.handle_event(event)
                drop_packet = event.metadata.get("packet")
                packet_id = (
                    drop_packet.packet_id
                    if drop_packet is not None
                    else event.packet_id
                )
                self._collector.record_drop(
                    current_time,
                    packet_id,
                    details={"reason": event.metadata.get("reason")},
                )
                return None

            return None

The collector and the existing
:class:`~events.history.EventHistoryLog` are complementary: the history
log captures the *engine* audit trail (event type, packet id, event
id), while the collector captures the *domain* audit trail (arrival,
service start, departure, drop) with the relevant context (drop
reason, service time).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterator


class MetricKind(Enum):
    """Categorical label for a :class:`MetricRecord`.

    Mirrors the four :class:`~events.types.EventType` values 1:1 so the
    collector can act as a pure event-to-record translator.
    """

    ARRIVAL = "arrival"
    SERVICE_START = "service_start"
    DEPARTURE = "departure"
    DROP = "drop"


@dataclass(frozen=True)
class MetricRecord:
    """A single entry in the :class:`MetricsCollector`.

    The record is frozen so a stored entry can never be mutated after
    capture — preserving the audit trail the collector is meant to
    provide.

    Attributes:
        timestamp: Simulation time at which the event was recorded.
        kind: Category of action recorded.
        packet_id: Identifier of the packet associated with the event,
            or ``None`` for events that are not packet-scoped.
        details: Free-form context for the record. The collector stores
            a defensive copy, so callers may mutate their input dict
            without affecting the stored record. Common keys include
            ``"reason"`` (for drops) and ``"service_time"`` (for
            departures), but the collector does not enforce or interpret
            them — that is the analytics layer's job.
    """

    timestamp: float
    kind: MetricKind
    packet_id: int | None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record.

        Returns:
            A dict with ``kind`` rendered as its string value so the
            result can be passed straight to :func:`json.dumps`.
        """
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


class MetricsCollector:
    """Append-only, computation-free recorder of simulation events.

    The collector stores every :class:`MetricRecord` it is asked to
    record and exposes simple inspection helpers (``__len__``,
    ``__iter__``, ``__getitem__``, :attr:`records`) so callers can
    slice, iterate, or count entries without copying.

    The collector's contract is to **store, not to compute**. It does
    not expose:

    * ``count_*`` methods (use ``len`` + filter, or the analytics layer)
    * ``average_*`` / ``mean_*`` / ``sum_*`` methods
    * ``drop_rate`` / ``throughput`` / ``utilization`` properties
    * ``summary`` / ``stats`` / ``report`` methods

    The :attr:`test_collector_does_not_expose_computed_methods` test
    guards against accidental drift toward computation.

    The collector is not an :class:`~simulation.engine.EventHandler`
    subclass; callers wire it into the engine by writing a small
    handler that dispatches each :class:`~events.types.EventType` to
    the matching ``record_*`` method. See the module docstring for a
    complete wiring example.

    The collector has no built-in size cap; callers that run long
    simulations should periodically clear it via :meth:`clear` to bound
    memory use.
    """

    def __init__(self) -> None:
        """Initialize an empty metrics collector."""
        self._records: list[MetricRecord] = []

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self._records)

    def __iter__(self) -> Iterator[MetricRecord]:
        """Iterate over recorded events in the order they were recorded."""
        return iter(self._records)

    def __getitem__(self, index: int | slice) -> MetricRecord | list[MetricRecord]:
        """Return a record by index or a list of records by slice.

        Parameters:
            index: Integer position or slice into the collector.

        Returns:
            The :class:`MetricRecord` at that position, or a list of
            records for a slice.
        """
        return self._records[index]

    @property
    def records(self) -> list[MetricRecord]:
        """Return a copy of the recorded entries as a list.

        Returns:
            A list of :class:`MetricRecord` in recording order.
            Mutating the returned list does not affect the collector.
        """
        return list(self._records)

    def record_arrival(
        self,
        timestamp: float,
        packet_id: int | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a packet arrival event.

        Parameters:
            timestamp: Simulation time of the arrival.
            packet_id: Identifier of the arrived packet. May be ``None``
                for events that are not packet-scoped.
            details: Optional free-form context to attach to the record.
        """
        self._records.append(
            MetricRecord(
                timestamp=timestamp,
                kind=MetricKind.ARRIVAL,
                packet_id=packet_id,
                details=dict(details) if details else {},
            )
        )

    def record_service_start(
        self,
        timestamp: float,
        packet_id: int | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a packet service-start event.

        Parameters:
            timestamp: Simulation time when service began.
            packet_id: Identifier of the packet entering service.
            details: Optional free-form context to attach to the record.
        """
        self._records.append(
            MetricRecord(
                timestamp=timestamp,
                kind=MetricKind.SERVICE_START,
                packet_id=packet_id,
                details=dict(details) if details else {},
            )
        )

    def record_departure(
        self,
        timestamp: float,
        packet_id: int | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a packet departure (service completion) event.

        Parameters:
            timestamp: Simulation time of the departure.
            packet_id: Identifier of the departed packet.
            details: Optional free-form context to attach to the record.
        """
        self._records.append(
            MetricRecord(
                timestamp=timestamp,
                kind=MetricKind.DEPARTURE,
                packet_id=packet_id,
                details=dict(details) if details else {},
            )
        )

    def record_drop(
        self,
        timestamp: float,
        packet_id: int | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a packet drop event.

        Parameters:
            timestamp: Simulation time of the drop.
            packet_id: Identifier of the dropped packet.
            details: Optional free-form context to attach to the record.
                Callers typically store the drop reason here, e.g.
                ``{"reason": "queue_full"}`` or
                ``{"reason": "baseline_drop"}``.
        """
        self._records.append(
            MetricRecord(
                timestamp=timestamp,
                kind=MetricKind.DROP,
                packet_id=packet_id,
                details=dict(details) if details else {},
            )
        )

    def clear(self) -> None:
        """Remove all recorded entries from the collector."""
        self._records.clear()

    def filter_by_kind(self, kind: MetricKind) -> list[MetricRecord]:
        """Return every record whose ``kind`` matches.

        Parameters:
            kind: Record category to filter on.

        Returns:
            Records, in recording order, whose ``kind`` matches.
        """
        return [record for record in self._records if record.kind is kind]

    def filter_by_packet(self, packet_id: int) -> list[MetricRecord]:
        """Return every record associated with ``packet_id``.

        Parameters:
            packet_id: Packet identifier to filter on.

        Returns:
            Records, in recording order, whose ``packet_id`` matches.
        """
        return [record for record in self._records if record.packet_id == packet_id]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return the collector as a list of plain dicts.

        Returns:
            A list of :class:`MetricRecord` data, with ``kind``
            rendered as its string value.
        """
        return [record.to_dict() for record in self._records]

    def to_json(self) -> str:
        """Serialize the collector to a JSON string.

        Returns:
            A JSON array of records, each rendered via
            :meth:`MetricRecord.to_dict`.
        """
        return json.dumps([record.to_dict() for record in self._records])
