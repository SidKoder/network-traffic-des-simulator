"""Packet domain model."""

from dataclasses import dataclass, field


@dataclass
class Packet:
    """Represents a network packet flowing through the simulation.

    Attributes:
        packet_id: Unique identifier for the packet.
        arrival_time: Simulation time when the packet arrived.
        service_start_time: Time service began (None if not yet served).
        departure_time: Time the packet left the system (None if in system).
        dropped: Whether the packet was dropped.
        drop_time: Simulation time of drop (None if not dropped).
        drop_reason: Human-readable reason for drop.
        size_bytes: Optional packet size for future extensions.
    """

    packet_id: int
    arrival_time: float
    service_start_time: float | None = None
    departure_time: float | None = None
    dropped: bool = False
    drop_time: float | None = None
    drop_reason: str | None = None
    size_bytes: int = field(default=0)

    @property
    def waiting_time(self) -> float | None:
        """Compute time spent waiting in queue before service.

        Returns:
            Waiting time, or None if service has not started.
        """
        if self.service_start_time is None:
            return None
        return self.service_start_time - self.arrival_time

    @property
    def service_time(self) -> float | None:
        """Compute time spent in service.

        Returns:
            Service duration, or None if not yet completed.
        """
        if self.service_start_time is None or self.departure_time is None:
            return None
        return self.departure_time - self.service_start_time

    @property
    def system_time(self) -> float | None:
        """Compute total time in system (arrival to departure).

        Returns:
            Total system time, or None if not yet departed.
        """
        if self.departure_time is None:
            return None
        return self.departure_time - self.arrival_time

    def mark_dropped(self, drop_time: float, reason: str) -> None:
        """Record that this packet was dropped.

        Parameters:
            drop_time: Simulation time of the drop event.
            reason: Explanation for the drop.
        """
        self.dropped = True
        self.drop_time = drop_time
        self.drop_reason = reason

    def mark_service_start(self, service_start_time: float) -> None:
        """Record when service began for this packet.

        Parameters:
            service_start_time: Simulation time service started.
        """
        self.service_start_time = service_start_time

    def mark_departure(self, departure_time: float) -> None:
        """Record when the packet departed the system.

        Parameters:
            departure_time: Simulation time of departure.
        """
        self.departure_time = departure_time
