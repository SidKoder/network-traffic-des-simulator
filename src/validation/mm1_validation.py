"""M/M/1 theoretical validation against simulator observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from analytics.system_statistics import compute_system_statistics
from config.models import QueueConfig, QueueDiscipline
from distributions.continuous import ExponentialDistribution
from distributions.poisson import HomogeneousPoissonProcess
from events.event import Event
from events.scheduler import EventScheduler
from events.types import EventType
from metrics.collector import MetricKind, MetricRecord, MetricsCollector
from queueing.manager import QueueManager
from simulation.engine import EventHandler, EventLoop
from simulation.router import Router
from simulation.server import Server
from validation.theoretical_metrics import mean_relative_error


@dataclass(frozen=True)
class MM1TheoreticalMetrics:
    """Closed-form steady-state M/M/1 metrics."""

    lambda_rate: float
    mu_rate: float
    rho: float
    utilization: float
    L: float
    Lq: float
    W: float
    Wq: float


@dataclass(frozen=True)
class MM1ObservedMetrics:
    """Observed M/M/1 metrics estimated from simulation output."""

    completed_packets: int
    throughput: float
    utilization: float
    L: float
    Lq: float
    W: float
    Wq: float


@dataclass(frozen=True)
class MM1MetricComparison:
    """Observed vs theoretical comparison for one metric."""

    name: str
    theoretical: float
    observed: float
    relative_error: float
    within_threshold: bool


@dataclass(frozen=True)
class MM1ValidationResult:
    """Full result bundle for one M/M/1 validation run."""

    theoretical: MM1TheoreticalMetrics
    observed: MM1ObservedMetrics
    threshold: float
    comparisons: tuple[MM1MetricComparison, ...]

    @property
    def passed(self) -> bool:
        """Return whether all tracked metrics are within threshold."""
        return all(comparison.within_threshold for comparison in self.comparisons)


def mm1_theoretical_metrics(lambda_rate: float, mu_rate: float) -> MM1TheoreticalMetrics:
    """Compute closed-form steady-state M/M/1 metrics."""
    if lambda_rate <= 0.0:
        raise ValueError("lambda_rate must be positive")
    if mu_rate <= 0.0:
        raise ValueError("mu_rate must be positive")
    if lambda_rate >= mu_rate:
        raise ValueError("M/M/1 theoretical validation requires lambda_rate < mu_rate")

    rho = lambda_rate / mu_rate
    one_minus_rho = 1.0 - rho
    L = rho / one_minus_rho
    Lq = (rho**2) / one_minus_rho
    W = 1.0 / (mu_rate - lambda_rate)
    Wq = rho / (mu_rate - lambda_rate)
    return MM1TheoreticalMetrics(
        lambda_rate=lambda_rate,
        mu_rate=mu_rate,
        rho=rho,
        utilization=rho,
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
    )


def run_mm1_validation(
    *,
    lambda_rate: float,
    mu_rate: float,
    simulation_time: float = 20_000.0,
    seed: int = 42,
    relative_error_threshold: float = 0.10,
) -> MM1ValidationResult:
    """Run an M/M/1 simulation and compare observed metrics to theory."""
    if simulation_time <= 0.0:
        raise ValueError("simulation_time must be positive")
    if relative_error_threshold < 0.0:
        raise ValueError("relative_error_threshold must be non-negative")

    theoretical = mm1_theoretical_metrics(lambda_rate, mu_rate)
    records = _simulate_mm1_records(
        lambda_rate=lambda_rate,
        mu_rate=mu_rate,
        simulation_time=simulation_time,
        seed=seed,
    )
    observed = _mm1_observed_metrics(records)

    pairs = (
        ("utilization", theoretical.utilization, observed.utilization),
        ("L", theoretical.L, observed.L),
        ("Lq", theoretical.Lq, observed.Lq),
        ("W", theoretical.W, observed.W),
        ("Wq", theoretical.Wq, observed.Wq),
    )
    comparisons = tuple(
        MM1MetricComparison(
            name=name,
            theoretical=th,
            observed=obs,
            relative_error=mean_relative_error(th, obs),
            within_threshold=mean_relative_error(th, obs) <= relative_error_threshold,
        )
        for name, th, obs in pairs
    )

    return MM1ValidationResult(
        theoretical=theoretical,
        observed=observed,
        threshold=relative_error_threshold,
        comparisons=comparisons,
    )


def format_mm1_validation_report(result: MM1ValidationResult) -> str:
    """Format a CLI-friendly M/M/1 validation report."""
    status = "PASS" if result.passed else "FAIL"
    lines = [
        (
            "M/M/1 Validation"
            f" (lambda={result.theoretical.lambda_rate:.3f},"
            f" mu={result.theoretical.mu_rate:.3f})"
        ),
        f"rho={result.theoretical.rho:.3f}, threshold={result.threshold:.1%}, status={status}",
        "",
        "Metric         Theoretical    Observed    Rel.Error    Within Threshold",
        "---------------------------------------------------------------------",
    ]
    for comparison in result.comparisons:
        marker = "yes" if comparison.within_threshold else "no"
        lines.append(
            f"{comparison.name:<13}"
            f"{comparison.theoretical:>11.4f}  "
            f"{comparison.observed:>9.4f}  "
            f"{comparison.relative_error:>9.2%}  "
            f"{marker:>16}"
        )

    lines.extend(
        [
            "",
            (
                "Observed support: "
                f"completed_packets={result.observed.completed_packets}, "
                f"throughput={result.observed.throughput:.4f}"
            ),
        ]
    )
    return "\n".join(lines)


def _simulate_mm1_records(
    *,
    lambda_rate: float,
    mu_rate: float,
    simulation_time: float,
    seed: int,
) -> list[MetricRecord]:
    rng = np.random.default_rng(seed)
    arrival_process = HomogeneousPoissonProcess(arrival_rate=lambda_rate, rng=rng)
    service_distribution = ExponentialDistribution(rate=mu_rate, rng=rng)

    queue = QueueManager(
        QueueConfig(capacity=None, queue_discipline=QueueDiscipline.FIFO)
    )
    server = Server()
    scheduler = EventScheduler()
    collector = MetricsCollector()
    router = Router(
        queue_manager=queue,
        server=server,
        scheduler=scheduler,
        service_time_distribution=service_distribution,
        baseline_drop_probability=0.0,
        rng=rng,
    )

    class _ValidationHandler(EventHandler):
        def handle(self, event: Event, current_time: float) -> Iterator[Event] | None:
            if event.event_type == EventType.PACKET_ARRIVAL:
                packet = router.handle_event(event)
                collector.record_arrival(current_time, packet.packet_id)
                next_ts = current_time + arrival_process.sample_inter_arrival_time()
                if next_ts <= simulation_time:
                    scheduler.schedule(next_ts, EventType.PACKET_ARRIVAL, packet_id=None)
                return None

            if event.event_type == EventType.PACKET_SERVICE_START:
                packet = router.handle_event(event)
                collector.record_service_start(
                    current_time, packet.packet_id, details=event.metadata
                )
                return None

            if event.event_type == EventType.PACKET_DEPARTURE:
                packet = router.handle_event(event)
                service_time = (
                    event.metadata.get("service_time")
                    if event.metadata is not None
                    else None
                )
                collector.record_departure(
                    current_time,
                    packet.packet_id,
                    details={"service_time": service_time},
                )
                return None

            if event.event_type == EventType.PACKET_DROP:
                router.handle_event(event)
                drop_packet = event.metadata.get("packet") if event.metadata else None
                packet_id = drop_packet.packet_id if drop_packet is not None else event.packet_id
                reason = event.metadata.get("reason") if event.metadata else None
                collector.record_drop(
                    current_time,
                    packet_id,
                    details={"reason": reason},
                )
                return None

            return None

        def on_start(self, current_time: float) -> None:
            first_ts = current_time + arrival_process.sample_inter_arrival_time()
            if first_ts <= simulation_time:
                scheduler.schedule(first_ts, EventType.PACKET_ARRIVAL, packet_id=None)

    loop = EventLoop(scheduler=scheduler, handler=_ValidationHandler())
    loop.run(max_time=simulation_time)
    return collector.records


def _mm1_observed_metrics(records: list[MetricRecord]) -> MM1ObservedMetrics:
    stats = compute_system_statistics(records)
    lifecycle = _completed_packet_lifecycle(records)

    completed_packets = len(lifecycle.system_times)
    mean_system_time = (
        float(np.mean(lifecycle.system_times)) if lifecycle.system_times else 0.0
    )
    mean_waiting_time = (
        float(np.mean(lifecycle.waiting_times)) if lifecycle.waiting_times else 0.0
    )

    return MM1ObservedMetrics(
        completed_packets=completed_packets,
        throughput=stats.success_rate_per_unit_time,
        utilization=stats.server_utilization_observed_tail,
        L=stats.average_queue_length + stats.server_utilization_observed_tail,
        Lq=stats.average_queue_length,
        W=mean_system_time,
        Wq=mean_waiting_time,
    )


@dataclass(frozen=True)
class _CompletedLifecycle:
    system_times: list[float]
    waiting_times: list[float]


def _completed_packet_lifecycle(records: list[MetricRecord]) -> _CompletedLifecycle:
    arrivals: dict[int, float] = {}
    starts: dict[int, float] = {}
    departures: dict[int, float] = {}

    for record in sorted(records, key=lambda r: r.timestamp):
        packet_id = record.packet_id
        if packet_id is None:
            continue
        if record.kind is MetricKind.ARRIVAL and packet_id not in arrivals:
            arrivals[packet_id] = record.timestamp
        elif record.kind is MetricKind.SERVICE_START and packet_id not in starts:
            starts[packet_id] = record.timestamp
        elif record.kind is MetricKind.DEPARTURE and packet_id not in departures:
            departures[packet_id] = record.timestamp

    system_times: list[float] = []
    waiting_times: list[float] = []
    for packet_id, departure_ts in departures.items():
        arrival_ts = arrivals.get(packet_id)
        if arrival_ts is None:
            continue
        system_times.append(departure_ts - arrival_ts)

        service_start_ts = starts.get(packet_id)
        if service_start_ts is not None:
            waiting_times.append(service_start_ts - arrival_ts)

    return _CompletedLifecycle(system_times=system_times, waiting_times=waiting_times)
