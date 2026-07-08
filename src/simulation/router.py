"""Router controller coordinating queue, server, scheduler, and distributions."""

import numpy as np

from distributions.base import Distribution
from distributions.discrete import BernoulliDistribution
from events.event import Event
from events.scheduler import EventScheduler
from events.types import EventType
from queueing.manager import QueueManager
from simulation.packet import Packet
from simulation.server import Server


class Router:
    """Controller for packet arrival, buffering, service, and drops."""

    def __init__(
        self,
        queue_manager: QueueManager,
        server: Server,
        scheduler: EventScheduler,
        service_time_distribution: Distribution,
        *,
        baseline_drop_probability: float = 0.0,
        rng: np.random.Generator | None = None,
        first_packet_id: int = 1,
    ) -> None:
        """Initialize a router controller.

        Parameters:
            queue_manager: Router memory used to buffer packets.
            server: Router CPU that services one packet at a time.
            scheduler: Event scheduler used for service/drop events.
            service_time_distribution: Distribution for service durations.
            baseline_drop_probability: Packet drop probability before queue checks.
            rng: Optional random generator for baseline drop checks.
            first_packet_id: Packet ID assigned to the first created packet.
        """
        if not 0.0 <= baseline_drop_probability <= 1.0:
            raise ValueError("baseline_drop_probability must be in [0, 1]")
        if first_packet_id < 0:
            raise ValueError("first_packet_id must be non-negative")

        self.queue_manager = queue_manager
        self.server = server
        self.scheduler = scheduler
        self.service_time_distribution = service_time_distribution
        self.baseline_drop_probability = baseline_drop_probability
        self._rng = rng if rng is not None else np.random.default_rng()
        self._baseline_drop_distribution = BernoulliDistribution(
            probability=baseline_drop_probability,
            rng=self._rng,
        )
        self._next_packet_id = first_packet_id
        self._packets_created = 0
        self._packets_dropped = 0
        self._service_start_pending = False

    @property
    def packets_created(self) -> int:
        """Return the number of packets created by this router."""
        return self._packets_created

    @property
    def packets_dropped(self) -> int:
        """Return the number of packets dropped by this router."""
        return self._packets_dropped

    def handle_event(self, event: Event) -> Packet | None:
        """Process a router-relevant event.

        Parameters:
            event: Event emitted by the simulation scheduler.

        Returns:
            Packet affected by the event, when the event produces one.
        """
        if event.event_type == EventType.PACKET_ARRIVAL:
            return self.handle_arrival(event.timestamp)
        if event.event_type == EventType.PACKET_SERVICE_START:
            return self.handle_service_start(event.timestamp, event.metadata)
        if event.event_type == EventType.PACKET_DEPARTURE:
            return self.handle_departure(event.timestamp)
        return None

    def handle_arrival(self, current_time: float) -> Packet:
        """Run the router arrival flow for a newly created packet.

        Flow:
            arrival -> baseline drop check -> queue full check -> enqueue ->
            if server idle, start service immediately; otherwise wait in queue.

        Parameters:
            current_time: Simulation time of the packet arrival.

        Returns:
            The packet created for this arrival.
        """
        packet = self._create_packet(current_time)

        if self._should_drop_baseline():
            self._drop_packet(packet, current_time, "baseline_drop")
            return packet

        if self._is_queue_in_overflow_state():
            self.queue_manager.record_drop()
            self._drop_packet(packet, current_time, "queue_full")
            return packet

        if not self.queue_manager.enqueue(packet):
            self._drop_packet(packet, current_time, "queue_full")
            return packet

        if not self.server.busy and not self._service_start_pending:
            self._schedule_service_start(current_time)

        return packet

    def handle_service_start(
        self,
        current_time: float,
        metadata: dict[str, object] | None = None,
    ) -> Packet:
        """Start CPU service and schedule the packet departure.

        Parameters:
            current_time: Simulation time when service begins.
            metadata: Event metadata containing the packet reserved for service.

        Returns:
            Packet whose service just started.
        """
        packet = self._packet_from_service_start_metadata(metadata)
        self._service_start_pending = False
        self.server.start_service(packet, current_time)

        service_time = self._sample_service_time()
        departure_time = current_time + service_time
        self.scheduler.schedule(
            departure_time,
            EventType.PACKET_DEPARTURE,
            packet_id=packet.packet_id,
            metadata={
                "packet": packet,
                "service_time": service_time,
            },
        )
        return packet

    def handle_departure(self, current_time: float) -> Packet:
        """Record departure, free the server, then inspect router memory.

        Flow:
            departure -> free server -> queue empty? -> stop or schedule next
            packet service start at the same simulation time.

        Parameters:
            current_time: Simulation time of the departure.

        Returns:
            Packet that departed the router CPU.
        """
        completed_packet = self.server.finish_service(departure_time=current_time)

        if self._queue_has_waiting_packet():
            self._schedule_service_start(current_time)

        return completed_packet

    def _create_packet(self, arrival_time: float) -> Packet:
        packet = Packet(packet_id=self._next_packet_id, arrival_time=arrival_time)
        self._next_packet_id += 1
        self._packets_created += 1
        return packet

    def _should_drop_baseline(self) -> bool:
        return bool(self._baseline_drop_distribution.sample(1)[0])

    def _is_queue_in_overflow_state(self) -> bool:
        return self.queue_manager.is_full

    def _drop_packet(self, packet: Packet, drop_time: float, reason: str) -> None:
        packet.mark_dropped(drop_time=drop_time, reason=reason)
        self._packets_dropped += 1
        self.scheduler.schedule(
            drop_time,
            EventType.PACKET_DROP,
            packet_id=packet.packet_id,
            metadata={"packet": packet, "reason": reason},
        )

    def _schedule_service_start(self, current_time: float) -> None:
        if self._service_start_pending:
            return

        packet = self.queue_manager.dequeue()
        self._service_start_pending = True

        self.scheduler.schedule(
            current_time,
            EventType.PACKET_SERVICE_START,
            packet_id=packet.packet_id,
            metadata={"packet": packet},
        )

    def _queue_has_waiting_packet(self) -> bool:
        return self.queue_manager.size > 0

    def _packet_from_service_start_metadata(
        self,
        metadata: dict[str, object] | None,
    ) -> Packet:
        # The packet was dequeued at scheduling time and stashed in metadata.
        # Always read it back from metadata; never dequeue here, otherwise
        # we'd pop the *next* packet in line and service the wrong one.
        if metadata is None:
            raise ValueError(
                "PACKET_SERVICE_START event arrived without metadata"
            )
        packet = metadata.get("packet")
        if not isinstance(packet, Packet):
            raise ValueError(
                "PACKET_SERVICE_START event metadata is missing a Packet"
            )
        return packet

    def _sample_service_time(self) -> float:
        sample = self.service_time_distribution.sample(1)[0]
        service_time = float(sample)
        if service_time < 0.0:
            raise ValueError("service_time_distribution produced a negative value")
        return service_time


RouterController = Router
