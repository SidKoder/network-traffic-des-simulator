"""Tests for centralized system-level statistics."""

from __future__ import annotations

import pytest

from analytics.system_statistics import (
    compute_system_statistics,
    compute_system_statistics_from_collector,
)
from metrics.collector import MetricKind, MetricRecord, MetricsCollector


def _r(timestamp: float, kind: MetricKind, packet_id: int | None) -> MetricRecord:
    return MetricRecord(timestamp=timestamp, kind=kind, packet_id=packet_id, details={})


def test_empty_records_return_zeroed_system_stats() -> None:
    stats = compute_system_statistics([])

    assert stats.total_arrivals == 0
    assert stats.total_departures == 0
    assert stats.total_drops == 0
    assert stats.accepted_packets == 0
    assert stats.observation_start_time is None
    assert stats.observation_end_time is None
    assert stats.observation_duration == 0.0
    assert stats.success_rate_per_unit_time == 0.0
    assert stats.drop_probability == 0.0
    assert stats.acceptance_rate == 0.0
    assert stats.completion_rate == 0.0
    assert stats.average_queue_length == 0.0
    assert stats.server_busy_intervals == ()
    assert stats.server_busy_time_closed == 0.0
    assert stats.server_busy_time_observed_tail == 0.0
    assert stats.server_utilization_closed_intervals == 0.0
    assert stats.server_utilization_observed_tail == 0.0


def test_basic_run_derives_throughput_and_ratios() -> None:
    records = [
        _r(0.0, MetricKind.ARRIVAL, 1),
        _r(0.0, MetricKind.SERVICE_START, 1),
        _r(2.0, MetricKind.DEPARTURE, 1),
    ]

    stats = compute_system_statistics(records)

    assert stats.total_arrivals == 1
    assert stats.total_departures == 1
    assert stats.total_drops == 0
    assert stats.accepted_packets == 1
    assert stats.observation_duration == pytest.approx(2.0)
    assert stats.success_rate_per_unit_time == pytest.approx(0.5)
    assert stats.drop_probability == pytest.approx(0.0)
    assert stats.acceptance_rate == pytest.approx(1.0)
    assert stats.completion_rate == pytest.approx(1.0)
    assert stats.average_queue_length == pytest.approx(0.0)
    assert stats.server_busy_time_closed == pytest.approx(2.0)
    assert stats.server_busy_time_observed_tail == pytest.approx(2.0)
    assert stats.server_utilization_closed_intervals == pytest.approx(1.0)
    assert stats.server_utilization_observed_tail == pytest.approx(1.0)
    assert len(stats.server_busy_intervals) == 1
    assert stats.server_busy_intervals[0].start_time == pytest.approx(0.0)
    assert stats.server_busy_intervals[0].end_time == pytest.approx(2.0)
    assert stats.server_busy_intervals[0].duration == pytest.approx(2.0)
    assert stats.server_busy_intervals[0].packet_id == 1


def test_time_weighted_average_queue_length_is_correct() -> None:
    # Timeline:
    # t=0: arrival(1), service_start(1) -> q=0
    # t=1: arrival(2) -> q=1 for [1,2)
    # t=2: arrival(3) -> q=2 for [2,4)
    # t=4: departure(1), service_start(2) -> q=1 for [4,8)
    # t=8: departure(2), service_start(3) -> q=0 for [8,12)
    # t=12: departure(3)
    # area = 1*1 + 2*2 + 1*4 + 0*4 = 9; duration = 12 => 0.75
    records = [
        _r(0.0, MetricKind.ARRIVAL, 1),
        _r(0.0, MetricKind.SERVICE_START, 1),
        _r(1.0, MetricKind.ARRIVAL, 2),
        _r(2.0, MetricKind.ARRIVAL, 3),
        _r(4.0, MetricKind.DEPARTURE, 1),
        _r(4.0, MetricKind.SERVICE_START, 2),
        _r(8.0, MetricKind.DEPARTURE, 2),
        _r(8.0, MetricKind.SERVICE_START, 3),
        _r(12.0, MetricKind.DEPARTURE, 3),
    ]

    stats = compute_system_statistics(records)

    assert stats.total_arrivals == 3
    assert stats.total_departures == 3
    assert stats.total_drops == 0
    assert stats.average_queue_length == pytest.approx(0.75)
    assert stats.success_rate_per_unit_time == pytest.approx(3.0 / 12.0)


def test_drop_probability_and_rates_with_mixed_outcomes() -> None:
    records = [
        _r(0.0, MetricKind.ARRIVAL, 1),
        _r(0.0, MetricKind.SERVICE_START, 1),
        _r(0.5, MetricKind.ARRIVAL, 2),
        _r(0.5, MetricKind.DROP, 2),
        _r(1.0, MetricKind.DEPARTURE, 1),
    ]

    stats = compute_system_statistics(records)

    assert stats.total_arrivals == 2
    assert stats.total_departures == 1
    assert stats.total_drops == 1
    assert stats.accepted_packets == 1
    assert stats.drop_probability == pytest.approx(0.5)
    assert stats.acceptance_rate == pytest.approx(0.5)
    assert stats.completion_rate == pytest.approx(0.5)
    assert stats.success_rate_per_unit_time == pytest.approx(1.0)


def test_records_are_sorted_by_timestamp_before_computation() -> None:
    ordered = [
        _r(0.0, MetricKind.ARRIVAL, 1),
        _r(0.0, MetricKind.SERVICE_START, 1),
        _r(1.0, MetricKind.DEPARTURE, 1),
    ]
    shuffled = [ordered[2], ordered[0], ordered[1]]

    expected = compute_system_statistics(ordered)
    actual = compute_system_statistics(shuffled)
    assert actual == expected


def test_inconsistent_stream_raises_when_queue_becomes_negative() -> None:
    records = [_r(1.0, MetricKind.SERVICE_START, 99)]

    with pytest.raises(ValueError, match="queue length became negative"):
        compute_system_statistics(records)


def test_collector_wrapper_uses_same_computation() -> None:
    collector = MetricsCollector()
    collector.record_arrival(0.0, packet_id=1)
    collector.record_service_start(0.0, packet_id=1)
    collector.record_departure(2.0, packet_id=1)

    stats = compute_system_statistics_from_collector(collector)
    assert stats.total_departures == 1
    assert stats.success_rate_per_unit_time == pytest.approx(0.5)


def test_open_tail_utilization_differs_from_closed_intervals() -> None:
    # Packet starts service at t=0 and remains in service until observation end t=5.
    # Closed busy time = 0 (no departure), observed-tail busy time = 5.
    records = [
        _r(0.0, MetricKind.ARRIVAL, 1),
        _r(0.0, MetricKind.SERVICE_START, 1),
        _r(5.0, MetricKind.ARRIVAL, 2),
    ]

    stats = compute_system_statistics(records)

    assert stats.observation_duration == pytest.approx(5.0)
    assert stats.server_busy_time_closed == pytest.approx(0.0)
    assert stats.server_busy_time_observed_tail == pytest.approx(5.0)
    assert stats.server_utilization_closed_intervals == pytest.approx(0.0)
    assert stats.server_utilization_observed_tail == pytest.approx(1.0)
    assert stats.server_busy_intervals == ()


def test_server_busy_intervals_are_stored_for_theoretical_validation() -> None:
    records = [
        _r(0.0, MetricKind.ARRIVAL, 1),
        _r(0.0, MetricKind.SERVICE_START, 1),
        _r(3.0, MetricKind.DEPARTURE, 1),
        _r(3.0, MetricKind.ARRIVAL, 2),
        _r(3.0, MetricKind.SERVICE_START, 2),
        _r(7.0, MetricKind.DEPARTURE, 2),
    ]

    stats = compute_system_statistics(records)

    assert len(stats.server_busy_intervals) == 2
    assert stats.server_busy_intervals[0].start_time == pytest.approx(0.0)
    assert stats.server_busy_intervals[0].end_time == pytest.approx(3.0)
    assert stats.server_busy_intervals[1].start_time == pytest.approx(3.0)
    assert stats.server_busy_intervals[1].end_time == pytest.approx(7.0)
    assert stats.server_busy_time_closed == pytest.approx(7.0)
    assert stats.server_busy_time_observed_tail == pytest.approx(7.0)
    assert stats.server_utilization_closed_intervals == pytest.approx(1.0)
    assert stats.server_utilization_observed_tail == pytest.approx(1.0)


def test_departure_without_active_service_raises() -> None:
    records = [_r(1.0, MetricKind.DEPARTURE, 1)]

    with pytest.raises(ValueError, match="departure without active service"):
        compute_system_statistics(records)
