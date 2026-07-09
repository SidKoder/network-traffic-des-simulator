"""Centralized packet-level latency computations.

This module is the single source of truth for packet timing formulas
used to quantify user-visible latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PacketLatencyStatistics:
    """Latency measures for a single packet."""

    packet_id: int
    waiting_time: float | None
    service_time: float | None
    total_delay_time: float | None


class PacketTimingLike(Protocol):
    """Protocol for packet-like timing inputs used by this module."""

    packet_id: int
    arrival_time: float
    service_start_time: float | None
    departure_time: float | None


def calculate_waiting_time(
    arrival_time: float,
    service_start_time: float | None,
) -> float | None:
    """Return queue waiting time (service_start - arrival)."""
    if service_start_time is None:
        return None
    return service_start_time - arrival_time


def calculate_service_time(
    service_start_time: float | None,
    departure_time: float | None,
) -> float | None:
    """Return service duration (departure - service_start)."""
    if service_start_time is None or departure_time is None:
        return None
    return departure_time - service_start_time


def calculate_total_delay_time(
    arrival_time: float,
    departure_time: float | None,
) -> float | None:
    """Return end-to-end packet delay (departure - arrival)."""
    if departure_time is None:
        return None
    return departure_time - arrival_time


def calculate_system_time(
    arrival_time: float,
    departure_time: float | None,
) -> float | None:
    """Compatibility alias for total delay time."""
    return calculate_total_delay_time(arrival_time, departure_time)


def compute_packet_latency_statistics(
    packet: PacketTimingLike,
) -> PacketLatencyStatistics:
    """Compute packet-level latency metrics for one packet."""
    return PacketLatencyStatistics(
        packet_id=packet.packet_id,
        waiting_time=calculate_waiting_time(
            packet.arrival_time,
            packet.service_start_time,
        ),
        service_time=calculate_service_time(
            packet.service_start_time,
            packet.departure_time,
        ),
        total_delay_time=calculate_total_delay_time(
            packet.arrival_time,
            packet.departure_time,
        ),
    )
