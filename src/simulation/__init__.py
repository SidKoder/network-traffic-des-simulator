"""Simulation module public API."""

from simulation.clock import SimulationClock
from simulation.engine import EventHandler, EventLoop, PrintEventHandler
from simulation.packet import Packet
from simulation.server import Server

__all__ = [
    "EventHandler",
    "EventLoop",
    "Packet",
    "PrintEventHandler",
    "Router",
    "RouterController",
    "Server",
    "SimulationClock",
]


def __getattr__(name: str) -> object:
    """Lazily expose router classes without creating import cycles."""
    if name in {"Router", "RouterController"}:
        from simulation.router import Router, RouterController

        return {"Router": Router, "RouterController": RouterController}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
