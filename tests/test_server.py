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


def test_finish_service_without_departure_time_returns_packet_untouched() -> None:
    """finish_service(departure_time=None) returns the packet without stamping it.

    The packet's departure_time is left as None so the engine can stamp it later
    or hand the packet off to a downstream component that records the time.
    """
    server = Server()
    packet = Packet(packet_id=1, arrival_time=2.0)
    server.start_service(packet, start_time=3.5)

    completed = server.finish_service()

    assert completed is packet
    assert packet.departure_time is None
    # Server is fully reset even when no departure time is provided.
    assert server.busy is False
    assert server.current_packet is None
    assert server.busy_start_time is None


def test_finish_service_preserves_service_start_time() -> None:
    """finish_service must not mutate the packet's service_start_time."""
    server = Server()
    packet = Packet(packet_id=1, arrival_time=2.0)
    server.start_service(packet, start_time=3.5)

    server.finish_service(departure_time=8.0)

    assert packet.service_start_time == 3.5
    # waiting_time is a function of service_start_time and arrival_time; it must
    # remain valid after departure.
    assert packet.waiting_time == pytest.approx(1.5)


def test_packet_lifecycle_properties_after_full_service() -> None:
    """After a full lifecycle, waiting_time, service_time, system_time are correct."""
    server = Server()
    packet = Packet(packet_id=42, arrival_time=2.0)

    server.start_service(packet, start_time=5.0)
    completed = server.finish_service(departure_time=9.0)

    assert completed is packet
    # Waited 3.0 in queue (arrived at 2.0, started at 5.0).
    assert packet.waiting_time == pytest.approx(3.0)
    # Served for 4.0 (started 5.0, departed 9.0).
    assert packet.service_time == pytest.approx(4.0)
    # Total time in system: 7.0.
    assert packet.system_time == pytest.approx(7.0)


def test_start_service_does_not_mutate_state_when_already_busy() -> None:
    """A failed start_service call must leave the original packet untouched.

    If the busy guard raises, the server's current_packet and busy_start_time
    must still reflect the first packet, and the second packet must remain
    unserviced (service_start_time is None).
    """
    server = Server()
    first = Packet(packet_id=1, arrival_time=0.0)
    second = Packet(packet_id=2, arrival_time=1.5)
    server.start_service(first, start_time=1.0)

    with pytest.raises(RuntimeError, match="server is busy"):
        server.start_service(second, start_time=2.0)

    # Original service state is preserved.
    assert server.busy is True
    assert server.current_packet is first
    assert server.busy_start_time == 1.0
    # Second packet was never registered.
    assert second.service_start_time is None


def test_finish_service_does_not_mutate_state_when_idle() -> None:
    """A failed finish_service call must not change any state."""
    server = Server()
    packet = Packet(packet_id=1, arrival_time=0.0)

    with pytest.raises(RuntimeError, match="server is idle"):
        server.finish_service(departure_time=1.0)

    # State is still the initial idle state.
    assert server.busy is False
    assert server.current_packet is None
    assert server.busy_start_time is None
    # The packet passed in is irrelevant and unchanged.
    assert packet.service_start_time is None
    assert packet.departure_time is None


def test_start_service_accepts_zero_and_equal_times() -> None:
    """Boundary: start_time == 0 and start_time == arrival_time are valid."""
    server = Server()
    packet = Packet(packet_id=1, arrival_time=0.0)

    server.start_service(packet, start_time=0.0)

    assert server.busy is True
    assert server.busy_start_time == 0.0
    assert packet.service_start_time == 0.0
    # No waiting time when service starts immediately on arrival.
    assert packet.waiting_time == pytest.approx(0.0)


def test_finish_service_with_zero_duration_service() -> None:
    """Boundary: departure_time == start_time is a valid zero-duration service."""
    server = Server()
    packet = Packet(packet_id=1, arrival_time=2.0)
    server.start_service(packet, start_time=5.0)

    completed = server.finish_service(departure_time=5.0)

    assert completed is packet
    assert packet.service_time == pytest.approx(0.0)
    assert server.busy is False


def test_start_service_uses_keyword_argument_start_time() -> None:
    """Defensive: callers must be able to pass start_time as a keyword.

    Locks in the public API signature so a future positional re-order doesn't
    silently break the engine.
    """
    server = Server()
    packet = Packet(packet_id=1, arrival_time=0.0)

    server.start_service(packet, start_time=4.0)

    assert packet.service_start_time == 4.0


@pytest.mark.parametrize(
    "arrival,start,departure,expected_wait,expected_service,expected_system",
    [
        (0.0, 0.0, 1.0, 0.0, 1.0, 1.0),
        (1.0, 2.0, 5.0, 1.0, 3.0, 4.0),
        (10.5, 12.25, 12.5, 1.75, 0.25, 2.0),
        (100.0, 100.0, 100.0, 0.0, 0.0, 0.0),
    ],
)
def test_packet_timing_properties_round_trip(
    arrival: float,
    start: float,
    departure: float,
    expected_wait: float,
    expected_service: float,
    expected_system: float,
) -> None:
    """A range of (arrival, start, departure) triples all produce correct metrics."""
    server = Server()
    packet = Packet(packet_id=1, arrival_time=arrival)
    server.start_service(packet, start_time=start)
    server.finish_service(departure_time=departure)

    assert packet.waiting_time == pytest.approx(expected_wait)
    assert packet.service_time == pytest.approx(expected_service)
    assert packet.system_time == pytest.approx(expected_system)


def test_non_monotonic_departure_documented_current_behavior() -> None:
    """Document the current behavior: server does NOT reject departure < start.

    The model currently allows a packet to "depart" before it "started serving".
    This produces a negative service_time. This test pins the current behavior
    so a future fix is intentional (e.g., adding a guard raises here).
    """
    server = Server()
    packet = Packet(packet_id=1, arrival_time=2.0)
    server.start_service(packet, start_time=10.0)

    completed = server.finish_service(departure_time=5.0)

    # Current (questionable) behavior: no validation, negative service_time.
    assert completed is packet
    assert packet.departure_time == 5.0
    assert packet.service_time == pytest.approx(-5.0)


def test_repeated_start_finish_cycles_keep_state_consistent() -> None:
    """After N cycles, server is idle and the last packet is correctly recorded."""
    server = Server()

    for i, (start, end) in enumerate([(0.0, 1.0), (2.0, 3.0), (4.0, 5.5)]):
        packet = Packet(packet_id=i, arrival_time=start)
        server.start_service(packet, start_time=start)
        assert server.busy is True
        assert server.current_packet is packet
        server.finish_service(departure_time=end)
        assert server.busy is False
        assert server.current_packet is None
        assert server.busy_start_time is None

    # Last cycle's packet has the right timing.
    last = Packet(packet_id=99, arrival_time=4.0)
    server.start_service(last, start_time=4.0)
    server.finish_service(departure_time=5.5)
    assert last.service_time == pytest.approx(1.5)


def test_two_packets_do_not_share_state() -> None:
    """Serving packet A then B must not leak any field from A into B."""
    server = Server()
    a = Packet(packet_id=1, arrival_time=0.0)
    b = Packet(packet_id=2, arrival_time=10.0)

    server.start_service(a, start_time=1.0)
    server.finish_service(departure_time=3.0)
    server.start_service(b, start_time=11.0)

    # b's service_start_time is its own, not a's.
    assert b.service_start_time == 11.0
    assert a.service_start_time == 1.0
    # b is the current packet; a is fully released.
    assert server.current_packet is b
    assert server.current_packet is not a

