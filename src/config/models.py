"""Configuration models for the DES engine."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class QueueDiscipline(str, Enum):
    """Supported queue service disciplines."""

    FIFO = "FIFO"
    LIFO = "LIFO"


class ArrivalConfig(BaseModel):
    """Parameters governing the packet arrival process.

    Attributes:
        arrival_rate: Poisson arrival rate (lambda), packets per unit time.
    """

    arrival_rate: float = Field(gt=0.0, description="Poisson arrival rate (lambda)")


class ServiceConfig(BaseModel):
    """Parameters governing packet service times.

    Attributes:
        service_rate: Exponential service rate (mu), packets per unit time.
    """

    service_rate: float = Field(gt=0.0, description="Exponential service rate (mu)")


class QueueConfig(BaseModel):
    """Parameters governing queue behavior.

    Attributes:
        capacity: Maximum queue size. ``None`` denotes infinite capacity (M/M/1).
        queue_discipline: Service order for waiting packets.
    """

    capacity: int | None = Field(
        default=None,
        ge=1,
        description="Queue capacity; None for infinite (M/M/1)",
    )
    queue_discipline: QueueDiscipline = QueueDiscipline.FIFO

    @field_validator("capacity")
    @classmethod
    def validate_capacity(cls, value: int | None) -> int | None:
        """Ensure finite capacity is at least 1.

        Parameters:
            value: Proposed queue capacity.

        Returns:
            Validated capacity value.
        """
        if value is not None and value < 1:
            raise ValueError("capacity must be >= 1 when specified")
        return value


class SimulationConfig(BaseModel):
    """Top-level simulation configuration.

    All simulation parameters must be supplied through this object.

    Attributes:
        arrival: Arrival process configuration.
        service: Service process configuration.
        queue: Queue configuration.
        simulation_time: Total simulated time horizon.
        random_seed: Optional seed for reproducible random streams.
    """

    arrival: ArrivalConfig
    service: ServiceConfig
    queue: QueueConfig
    simulation_time: float = Field(gt=0.0, description="Simulation horizon")
    random_seed: int | None = Field(default=None, description="RNG seed")

    @property
    def traffic_intensity(self) -> float:
        """Compute traffic intensity rho = lambda / mu.

        Returns:
            Ratio of arrival rate to service rate.
        """
        return self.arrival.arrival_rate / self.service.service_rate

    @property
    def is_stable(self) -> bool:
        """Check M/M/1 stability condition (rho < 1 for infinite queue).

        Returns:
            True when the system is stable under infinite-buffer assumption.
        """
        if self.queue.capacity is not None:
            return True
        return self.traffic_intensity < 1.0
