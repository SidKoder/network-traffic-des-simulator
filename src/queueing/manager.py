"""Queue management for M/M/1 and M/M/1/K systems."""

from collections import deque

from config.models import QueueConfig, QueueDiscipline
from simulation.packet import Packet


class QueueFullError(Exception):
    """Raised when enqueue is attempted on a full finite-capacity queue."""


class QueueManager:
    """Manages packet buffering with configurable capacity and discipline.

    Supports infinite-capacity M/M/1 (capacity=None) and finite M/M/1/K queues.
    """

    def __init__(self, config: QueueConfig) -> None:
        """Initialize the queue manager.

        Parameters:
            config: Queue configuration including capacity and discipline.
        """
        self._config = config
        self._queue: deque[Packet] = deque()
        self._total_enqueued: int = 0
        self._total_dropped: int = 0

    @property
    def config(self) -> QueueConfig:
        """Return the queue configuration.

        Returns:
            QueueConfig instance.
        """
        return self._config

    @property
    def capacity(self) -> int | None:
        """Return queue capacity (None for infinite).

        Returns:
            Maximum queue size or None.
        """
        return self._config.capacity

    @property
    def size(self) -> int:
        """Return current number of packets in the queue.

        Returns:
            Current queue length.
        """
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """Check whether the queue has reached capacity.

        Returns:
            True if finite queue is at capacity, False otherwise.
        """
        if self._config.capacity is None:
            return False
        return len(self._queue) >= self._config.capacity

    @property
    def is_empty(self) -> bool:
        """Check whether the queue contains no packets.

        Returns:
            True if queue is empty.
        """
        return len(self._queue) == 0

    @property
    def total_enqueued(self) -> int:
        """Return total packets successfully enqueued.

        Returns:
            Cumulative enqueue count.
        """
        return self._total_enqueued

    @property
    def total_dropped(self) -> int:
        """Return total packets dropped due to capacity.

        Returns:
            Cumulative drop count.
        """
        return self._total_dropped

    def enqueue(self, packet: Packet) -> bool:
        """Attempt to add a packet to the queue.

        For finite-capacity queues, returns False and increments drop count
        when the queue is full. Infinite queues always accept packets.

        Parameters:
            packet: Packet to enqueue.

        Returns:
            True if enqueued, False if dropped due to capacity.
        """
        if self.is_full:
            self._total_dropped += 1
            return False

        if self._config.queue_discipline == QueueDiscipline.FIFO:
            self._queue.append(packet)
        elif self._config.queue_discipline == QueueDiscipline.LIFO:
            self._queue.appendleft(packet)

        self._total_enqueued += 1
        return True

    def dequeue(self) -> Packet:
        """Remove and return the next packet for service.

        Returns:
            Next packet according to queue discipline.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty:
            raise IndexError("Cannot dequeue from empty queue")
        return self._queue.popleft()

    def peek(self) -> Packet | None:
        """Inspect the next packet without removing it.

        Returns:
            Head packet for FIFO, or None if empty.
        """
        if self.is_empty:
            return None
        return self._queue[0]

    def clear(self) -> None:
        """Remove all packets and reset counters."""
        self._queue.clear()
        self._total_enqueued = 0
        self._total_dropped = 0
