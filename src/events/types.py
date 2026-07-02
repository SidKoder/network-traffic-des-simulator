"""Event type definitions for the DES engine."""

from enum import Enum


class EventType(Enum):
    """Supported discrete simulation events."""

    PACKET_ARRIVAL = "packet_arrival"
    PACKET_SERVICE_START = "packet_service_start"
    PACKET_DEPARTURE = "packet_departure"
    PACKET_DROP = "packet_drop"


def __getattr__(name: str) -> object:
    """Preserve the former ``events.types.Event`` import path."""
    if name == "Event":
        from events.event import Event

        return Event
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
