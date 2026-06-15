"""Event type definitions for the DES engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """Supported discrete simulation events."""

    PACKET_ARRIVAL = "packet_arrival"
    PACKET_SERVICE_START = "packet_service_start"
    PACKET_DEPARTURE = "packet_departure"
    PACKET_DROP = "packet_drop"


@dataclass(order=True)
class Event:
    """A scheduled simulation event.

    Events are ordered by timestamp for priority-queue scheduling.
    A sequence number breaks ties when timestamps are equal.

    Attributes:
        timestamp: Simulation time at which the event occurs.
        sequence: Tie-breaker for events with identical timestamps.
        event_type: Category of the event.
        payload: Arbitrary event-associated data.
    """

    timestamp: float
    sequence: int = field(compare=True)
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)
