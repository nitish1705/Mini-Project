"""
Performance metrics for FANET routing evaluation.

Computes:
  • Packet Delivery Ratio (PDR)
  • End-to-End Delay
  • Network Throughput
  • Energy Consumption
  • Network Lifetime
"""

from typing import Any, Dict, List

import numpy as np


class MetricsCollector:
    """Accumulates per-episode metrics and computes aggregates."""

    def __init__(self):
        self.episodes: List[Dict[str, float]] = []

    def record(self, info: Dict[str, Any]):
        """Record metrics from one completed episode."""
        self.episodes.append(
            {
                "pdr": info.get("pdr", 0.0),
                "avg_delay": info.get("avg_delay", 0.0),
                "energy_used": info.get("energy_used", 0.0),
                "alive_nodes": info.get("alive_nodes", 0),
                "packets_sent": info.get("packets_sent", 0),
                "packets_received": info.get("packets_received", 0),
            }
        )

    def summary(self) -> Dict[str, float]:
        """Return averaged metrics across all recorded episodes."""
        if not self.episodes:
            return {}
        keys = self.episodes[0].keys()
        return {k: np.mean([ep[k] for ep in self.episodes]) for k in keys}

    def reset(self):
        self.episodes.clear()


def packet_delivery_ratio(sent: int, received: int) -> float:
    """PDR = packets_received / packets_sent."""
    return received / max(1, sent)


def average_end_to_end_delay(delays: List[float]) -> float:
    """Average packet travel time."""
    return float(np.mean(delays)) if delays else 0.0


def network_throughput(
    total_bits_delivered: float, duration_seconds: float
) -> float:
    """Total bits delivered per second."""
    return total_bits_delivered / max(1e-6, duration_seconds)


def average_energy_consumption(energies_used: List[float]) -> float:
    """Average battery usage across UAV nodes."""
    return float(np.mean(energies_used)) if energies_used else 0.0


def network_lifetime(residual_energies: List[float], drain_rate: float) -> float:
    """
    Estimated time (seconds) until first node battery depletion.

    Uses the node with the lowest residual energy divided by
    the average drain rate.
    """
    if not residual_energies or drain_rate <= 0:
        return float("inf")
    min_energy = min(residual_energies)
    return min_energy / drain_rate


def compare_methods(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Given a dict of {method_name: metrics_dict}, return the same
    structure — handy as a passthrough for the plotting module.
    """
    return results
