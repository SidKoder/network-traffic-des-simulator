"""Tests for packet lifecycle timestamp storage."""

from simulation.packet import Packet


class TestPacketLifecycle:
    """Tests for Packet timestamp transitions."""

    def test_service_lifecycle_timestamps_are_stored(self) -> None:
        """Arrival, service start, and departure timestamps are preserved."""
        packet = Packet(packet_id=101, arrival_time=1.25)

        assert packet.arrival_time == 1.25
        assert packet.service_start_time is None
        assert packet.departure_time is None
        assert packet.drop_time is None
        assert packet.dropped is False

        packet.mark_service_start(service_start_time=2.75)
        packet.mark_departure(departure_time=6.5)

        assert packet.arrival_time == 1.25
        assert packet.service_start_time == 2.75
        assert packet.departure_time == 6.5
        assert packet.drop_time is None
        assert packet.dropped is False

    def test_drop_lifecycle_timestamp_is_stored(self) -> None:
        """Drop timestamp and reason are preserved when a packet is dropped."""
        packet = Packet(packet_id=202, arrival_time=3.0)

        packet.mark_dropped(drop_time=4.125, reason="queue_full")

        assert packet.arrival_time == 3.0
        assert packet.drop_time == 4.125
        assert packet.drop_reason == "queue_full"
        assert packet.dropped is True
        assert packet.service_start_time is None
        assert packet.departure_time is None
