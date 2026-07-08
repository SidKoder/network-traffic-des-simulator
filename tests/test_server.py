"""Tests for the router CPU server model."""

import pytest

from simulation.packet import Packet
from simulation.server import Server


def test_server_starts_idle() -> None:
    """A new router CPU has no packet in service."""
    server = Server()

    assert server.busy is False
    assert server.current_packet is None
    assert server.busy_start_time is None


def test_start_service_moves_server_to_busy_state() -> None:
    """start_service claims the CPU and records service state."""
    server = Server()
    packet = Packet(packet_id=1, arrival_time=2.0)

    server.start_service(packet, start_time=3.5)

    assert server.busy is True
    assert server.current_packet is packet
    assert server.busy_start_time == 3.5
    assert packet.service_start_time == 3.5


def test_finish_service_moves_server_back_to_idle_state() -> None:
    """finish_service releases the CPU and returns the completed packet."""
    server = Server()
    packet = Packet(packet_id=1, arrival_time=2.0)
    server.start_service(packet, start_time=3.5)

    completed = server.finish_service(departure_time=8.0)

    assert completed is packet
    assert packet.departure_time == 8.0
    assert server.busy is False
    assert server.current_packet is None
    assert server.busy_start_time is None


def test_busy_start_finish_transition_sequence() -> None:
    """The expected idle -> busy -> idle transition is maintained."""
    server = Server()
    packet = Packet(packet_id=10, arrival_time=1.0)

    assert server.busy is False

    server.start_service(packet, start_time=1.25)
    assert server.busy is True

    server.finish_service(departure_time=2.75)
    assert server.busy is False


def test_start_service_raises_when_server_is_busy() -> None:
    """A busy router CPU cannot accept another packet."""
    server = Server()
    server.start_service(Packet(packet_id=1, arrival_time=0.0), start_time=1.0)

    with pytest.raises(RuntimeError, match="server is busy"):
        server.start_service(Packet(packet_id=2, arrival_time=1.5), start_time=2.0)


def test_finish_service_raises_when_server_is_idle() -> None:
    """An idle router CPU cannot finish service."""
    server = Server()

    with pytest.raises(RuntimeError, match="server is idle"):
        server.finish_service(departure_time=1.0)

