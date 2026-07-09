"""Centralized system-level performance statistics.

This module computes whole-run network KPIs from recorded metric events.
Formulas live here (analytics layer), never in engine/router/collector code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from metrics.collector import MetricKind, MetricRecord, MetricsCollector


@dataclass(frozen=True)
class ServerBusyInterval:
    """One closed interval where the server was continuously busy."""

    start_time: float
    end_time: float
    duration: float
    packet_id: int | None


@dataclass(frozen=True)
class SystemStatistics:
    """System-level statistics for one simulation run."""

    total_arrivals: int
    total_departures: int
    total_drops: int
    accepted_packets: int
    observation_start_time: float | None
    observation_end_time: float | None
    observation_duration: float
    success_rate_per_unit_time: float
    drop_probability: float
    acceptance_rate: float
    completion_rate: float
    average_queue_length: float
    server_busy_intervals: tuple[ServerBusyInterval, ...]
    server_busy_time_closed: float
    server_busy_time_observed_tail: float
    server_utilization_closed_intervals: float
    server_utilization_observed_tail: float


def compute_system_statistics(records: Sequence[MetricRecord]) -> SystemStatistics:
    """Compute centralized run-level KPIs from metric records.

    Metrics are derived from the event stream only:
      - success_rate_per_unit_time = departures / observation_duration
      - drop_probability = drops / arrivals
      - acceptance_rate = accepted / arrivals
      - completion_rate = departures / arrivals
      - average_queue_length = time-weighted queue-length mean
      - server utilization from reconstructed SERVICE_START/DEPARTURE intervals
    """
    if not records:
        return SystemStatistics(
            total_arrivals=0,
            total_departures=0,
            total_drops=0,
            accepted_packets=0,
            observation_start_time=None,
            observation_end_time=None,
            observation_duration=0.0,
            success_rate_per_unit_time=0.0,
            drop_probability=0.0,
            acceptance_rate=0.0,
            completion_rate=0.0,
            average_queue_length=0.0,
            server_busy_intervals=(),
            server_busy_time_closed=0.0,
            server_busy_time_observed_tail=0.0,
            server_utilization_closed_intervals=0.0,
            server_utilization_observed_tail=0.0,
        )

    ordered = sorted(records, key=lambda r: r.timestamp)
    first_timestamp = ordered[0].timestamp
    last_timestamp = ordered[-1].timestamp
    duration = last_timestamp - first_timestamp

    total_arrivals = sum(1 for record in ordered if record.kind is MetricKind.ARRIVAL)
    total_departures = sum(
        1 for record in ordered if record.kind is MetricKind.DEPARTURE
    )
    total_drops = sum(1 for record in ordered if record.kind is MetricKind.DROP)

    dropped_ids = {
        record.packet_id
        for record in ordered
        if record.kind is MetricKind.DROP and record.packet_id is not None
    }
    accepted_packets = sum(
        1
        for record in ordered
        if record.kind is MetricKind.ARRIVAL
        and record.packet_id is not None
        and record.packet_id not in dropped_ids
    )

    average_queue_length = _time_weighted_average_queue_length(ordered, dropped_ids)
    server_profile = _derive_server_busy_profile(ordered, last_timestamp)

    success_rate_per_unit_time = (
        total_departures / duration if duration > 0.0 else 0.0
    )
    drop_probability = total_drops / total_arrivals if total_arrivals > 0 else 0.0
    acceptance_rate = (
        accepted_packets / total_arrivals if total_arrivals > 0 else 0.0
    )
    completion_rate = (
        total_departures / total_arrivals if total_arrivals > 0 else 0.0
    )
    server_utilization_closed_intervals = (
        server_profile.closed_busy_time / duration if duration > 0.0 else 0.0
    )
    server_utilization_observed_tail = (
        server_profile.observed_tail_busy_time / duration if duration > 0.0 else 0.0
    )

    return SystemStatistics(
        total_arrivals=total_arrivals,
        total_departures=total_departures,
        total_drops=total_drops,
        accepted_packets=accepted_packets,
        observation_start_time=first_timestamp,
        observation_end_time=last_timestamp,
        observation_duration=duration,
        success_rate_per_unit_time=success_rate_per_unit_time,
        drop_probability=drop_probability,
        acceptance_rate=acceptance_rate,
        completion_rate=completion_rate,
        average_queue_length=average_queue_length,
        server_busy_intervals=tuple(server_profile.closed_intervals),
        server_busy_time_closed=server_profile.closed_busy_time,
        server_busy_time_observed_tail=server_profile.observed_tail_busy_time,
        server_utilization_closed_intervals=server_utilization_closed_intervals,
        server_utilization_observed_tail=server_utilization_observed_tail,
    )


def compute_system_statistics_from_collector(
    collector: MetricsCollector,
) -> SystemStatistics:
    """Compute system-level KPIs from a MetricsCollector."""
    return compute_system_statistics(collector.records)


def _time_weighted_average_queue_length(
    ordered: Sequence[MetricRecord],
    dropped_ids: set[int],
) -> float:
    if not ordered:
        return 0.0

    queue_length = 0
    area = 0.0
    previous_time = ordered[0].timestamp

    for record in ordered:
        delta_t = record.timestamp - previous_time
        if delta_t < 0:
            raise ValueError("Metric records must be non-decreasing by timestamp")
        area += queue_length * delta_t
        previous_time = record.timestamp

        if record.kind is MetricKind.ARRIVAL:
            if record.packet_id is not None and record.packet_id not in dropped_ids:
                queue_length += 1
        elif record.kind is MetricKind.SERVICE_START:
            queue_length -= 1
            if queue_length < 0:
                raise ValueError(
                    "Inconsistent metrics stream: queue length became negative"
                )

    duration = ordered[-1].timestamp - ordered[0].timestamp
    if duration <= 0.0:
        return 0.0
    return area / duration


@dataclass(frozen=True)
class _ServerBusyProfile:
    closed_intervals: list[ServerBusyInterval]
    closed_busy_time: float
    observed_tail_busy_time: float


def _derive_server_busy_profile(
    ordered: Sequence[MetricRecord],
    last_timestamp: float,
) -> _ServerBusyProfile:
    closed_intervals: list[ServerBusyInterval] = []
    current_start: float | None = None
    current_packet_id: int | None = None
    closed_busy_time = 0.0

    for record in ordered:
        if record.kind is MetricKind.SERVICE_START:
            if current_start is not None:
                raise ValueError(
                    "Inconsistent metrics stream: service started while server busy"
                )
            current_start = record.timestamp
            current_packet_id = record.packet_id
            continue

        if record.kind is MetricKind.DEPARTURE:
            if current_start is None:
                raise ValueError(
                    "Inconsistent metrics stream: departure without active service"
                )
            if (
                current_packet_id is not None
                and record.packet_id is not None
                and current_packet_id != record.packet_id
            ):
                raise ValueError(
                    "Inconsistent metrics stream: departure packet does not match active service"
                )
            busy_duration = record.timestamp - current_start
            if busy_duration < 0.0:
                raise ValueError("Inconsistent metrics stream: negative busy interval")

            closed_busy_time += busy_duration
            closed_intervals.append(
                ServerBusyInterval(
                    start_time=current_start,
                    end_time=record.timestamp,
                    duration=busy_duration,
                    packet_id=current_packet_id
                    if current_packet_id is not None
                    else record.packet_id,
                )
            )
            current_start = None
            current_packet_id = None

    observed_tail_busy_time = closed_busy_time
    if current_start is not None:
        tail_duration = last_timestamp - current_start
        if tail_duration < 0.0:
            raise ValueError(
                "Inconsistent metrics stream: open busy interval after observation end"
            )
        observed_tail_busy_time += tail_duration

    return _ServerBusyProfile(
        closed_intervals=closed_intervals,
        closed_busy_time=closed_busy_time,
        observed_tail_busy_time=observed_tail_busy_time,
    )
