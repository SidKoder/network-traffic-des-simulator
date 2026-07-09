"""Tests for centralized packet-level latency computations."""

from __future__ import annotations

from simulation.packet import Packet

from analytics.packet_statistics import (
    PacketLatencyStatistics,
    calculate_service_time,
    calculate_system_time,
    calculate_total_delay_time,
    calculate_waiting_time,
    compute_packet_latency_statistics,
)


def test_waiting_service_and_total_delay_formulas_are_centralized() -> None:
    """Core latency formulas return expected values for a complete lifecycle."""
    assert calculate_waiting_time(arrival_time=2.0, service_start_time=5.0) == 3.0
    assert calculate_service_time(service_start_time=5.0, departure_time=9.0) == 4.0
    assert calculate_total_delay_time(arrival_time=2.0, departure_time=9.0) == 7.0
    assert calculate_system_time(arrival_time=2.0, departure_time=9.0) == 7.0


def test_latency_formulas_return_none_for_incomplete_lifecycle() -> None:
    """Missing timestamps yield None for the corresponding derived metric."""
    assert calculate_waiting_time(arrival_time=2.0, service_start_time=None) is None
    assert calculate_service_time(service_start_time=None, departure_time=9.0) is None
    assert calculate_service_time(service_start_time=5.0, departure_time=None) is None
    assert calculate_total_delay_time(arrival_time=2.0, departure_time=None) is None
    assert calculate_system_time(arrival_time=2.0, departure_time=None) is None


def test_compute_packet_latency_statistics_for_packet() -> None:
    """Packet-level statistics are computed consistently from packet fields."""
    packet = Packet(packet_id=42, arrival_time=2.0)
    packet.mark_service_start(5.0)
    packet.mark_departure(9.0)

    stats = compute_packet_latency_statistics(packet)

    assert stats == PacketLatencyStatistics(
        packet_id=42,
        waiting_time=3.0,
        service_time=4.0,
        total_delay_time=7.0,
    )


def test_packet_properties_delegate_to_centralized_analytics_formulas() -> None:
    """Packet properties preserve behavior while delegating formula ownership."""
    packet = Packet(packet_id=1, arrival_time=10.0)
    assert packet.waiting_time is None
    assert packet.service_time is None
    assert packet.system_time is None

    packet.mark_service_start(12.5)
    packet.mark_departure(20.0)

    assert packet.waiting_time == 2.5
    assert packet.service_time == 7.5
    assert packet.system_time == 10.0
