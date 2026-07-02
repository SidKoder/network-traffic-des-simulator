"""Core event model for the discrete-event simulator."""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from events.types import EventType


@dataclass(order=True)
class Event:
    """A future action scheduled in the simulator.

    Events compare chronologically by ``timestamp``. The scheduler-assigned
    ``sequence`` is used as a deterministic FIFO tie-breaker when two events
    have the same timestamp. All descriptive fields are excluded from
    comparisons so their values never affect scheduling order.

    Attributes:
        timestamp: Simulation time at which the action occurs.
        event_type: Category of action to perform.
        packet_id: Identifier of the packet associated with the action, if any.
        metadata: Additional action-specific context.
        event_id: Globally unique identifier for this event.
        sequence: Scheduler-assigned FIFO tie-breaker.
    """

    timestamp: float
    event_type: EventType = field(compare=False)
    packet_id: int | None = field(default=None, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    event_id: str = field(default_factory=lambda: str(uuid4()), compare=False)
    sequence: int = field(default=0, compare=True)

    @property
    def payload(self) -> dict[str, Any]:
        """Return metadata under the scheduler's legacy attribute name."""
        return self.metadata
