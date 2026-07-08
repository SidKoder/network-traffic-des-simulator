"""Server model representing the router CPU."""

from simulation.packet import Packet


class Server:
    """Track router CPU state while one packet is in service."""

    def __init__(self) -> None:
        """Initialize an idle server."""
        self.busy: bool = False
        self.current_packet: Packet | None = None
        self.busy_start_time: float | None = None

    def start_service(self, packet: Packet, start_time: float) -> None:
        """Start servicing a packet.

        Parameters:
            packet: Packet assigned to the router CPU.
            start_time: Simulation time when service begins.

        Raises:
            RuntimeError: If the server is already busy.
        """
        if self.busy:
            raise RuntimeError("Cannot start service while server is busy")

        self.busy = True
        self.current_packet = packet
        self.busy_start_time = start_time
        packet.mark_service_start(start_time)

    def finish_service(self, departure_time: float | None = None) -> Packet:
        """Finish servicing the current packet and return it.

        Parameters:
            departure_time: Optional simulation time when service completes.

        Returns:
            Packet that just completed service.

        Raises:
            RuntimeError: If the server is idle.
        """
        if not self.busy or self.current_packet is None:
            raise RuntimeError("Cannot finish service while server is idle")

        packet = self.current_packet
        if departure_time is not None:
            packet.mark_departure(departure_time)

        self.busy = False
        self.current_packet = None
        self.busy_start_time = None
        return packet

