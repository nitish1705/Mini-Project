"""
FANET simulation environment with Gymnasium-compatible interface.

This module provides a pure-Python simulation that mirrors the NS-3
parameters (50 UAVs, 2000x2000 area, Random Waypoint mobility, etc.)
and exposes an OpenAI Gym-style API so the RL agent can train against
it without requiring a live NS-3 installation.

If NS-3 + ns3-gym is available on the system, the `NS3Bridge` helper
class at the bottom can be used to drive a real NS-3 simulation instead.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation.uav_node import UAVNode, UAVSwarm
from graph.graph_builder import GraphBuilder


class FANETEnv:
    """Gymnasium-style environment wrapping the FANET simulation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.num_nodes: int = cfg.get("num_nodes", 50)
        self.area_width: float = cfg.get("area_width", 2000.0)
        self.area_height: float = cfg.get("area_height", 2000.0)
        self.comm_range: float = cfg.get("comm_range", 250.0)
        self.max_speed: float = cfg.get("max_speed", 15.0)
        self.max_steps: int = cfg.get("max_steps", 100)  # 100 s
        self.packet_size: int = cfg.get("packet_size", 512)
        self.num_multicast_dests: int = cfg.get("num_multicast_dests", 5)

        self.swarm: Optional[UAVSwarm] = None
        self.graph_builder = GraphBuilder(comm_range=self.comm_range)
        self.current_step: int = 0

        # Per-episode counters for metrics
        self.packets_sent: int = 0
        self.packets_received: int = 0
        self.total_delay: float = 0.0
        self.total_energy_used: float = 0.0
        self._initial_energies: List[float] = []

        # Current multicast session
        self.source: int = 0
        self.destinations: List[int] = []

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
        self.swarm = UAVSwarm(
            num_nodes=self.num_nodes,
            area_width=self.area_width,
            area_height=self.area_height,
            comm_range=self.comm_range,
            max_speed=self.max_speed,
        )
        self.current_step = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.total_delay = 0.0
        self.total_energy_used = 0.0
        self._initial_energies = [n.residual_energy for n in self.swarm.nodes]

        # Choose random multicast session
        alive_ids = [n.node_id for n in self.swarm.get_alive_nodes()]
        self.source = random.choice(alive_ids)
        remaining = [i for i in alive_ids if i != self.source]
        k = min(self.num_multicast_dests, len(remaining))
        self.destinations = random.sample(remaining, k)

        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Execute one routing decision step.

        *action* is a dict produced by the routing engine:
            {
                "relay_nodes": [list of relay node ids],
                "paths": {dest_id: [ordered node path], ...}
            }

        Returns (obs, reward, done, info).
        """
        assert self.swarm is not None, "Call reset() first"

        relay_nodes: List[int] = action.get("relay_nodes", [])
        paths: Dict[int, List[int]] = action.get("paths", {})

        step_delivered = 0
        step_delay = 0.0
        step_energy = 0.0

        for dest, path in paths.items():
            self.packets_sent += 1
            delivered, delay, energy = self._simulate_packet(path)
            if delivered:
                self.packets_received += 1
                step_delivered += 1
            step_delay += delay
            step_energy += energy

        # Advance mobility
        self.swarm.step(dt=1.0)
        self.current_step += 1

        self.total_delay += step_delay
        self.total_energy_used += step_energy

        done = self.current_step >= self.max_steps or self.swarm.num_alive < 2

        reward = self._compute_reward(step_delivered, step_delay, step_energy, len(paths))

        obs = self._get_obs()
        info = {
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "pdr": self.packets_received / max(1, self.packets_sent),
            "avg_delay": self.total_delay / max(1, self.packets_received),
            "energy_used": self.total_energy_used,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_packet(self, path: List[int]) -> Tuple[bool, float, float]:
        """
        Simulate forwarding a packet along *path*.
        Returns (delivered, delay, energy_consumed).
        """
        delay = 0.0
        energy = 0.0

        for i in range(len(path) - 1):
            src_node = self.swarm.get_node(path[i])
            dst_node = self.swarm.get_node(path[i + 1])

            if not src_node.is_alive or not dst_node.is_alive:
                return False, delay, energy

            dist = src_node.distance_to(dst_node)
            if dist > self.comm_range:
                return False, delay, energy  # link broken

            # Simple propagation model
            hop_delay = 0.001 + dist / 3e8 + random.uniform(0, 0.005)  # seconds
            loss_prob = min(0.5, (dist / self.comm_range) ** 2 * 0.3)

            if random.random() < loss_prob:
                return False, delay + hop_delay, energy

            src_node.consume_tx_energy()
            dst_node.consume_rx_energy()
            delay += hop_delay
            energy += src_node._energy_tx_cost + dst_node._energy_rx_cost

        return True, delay, energy

    def _compute_reward(
        self, delivered: int, delay: float, energy: float, num_paths: int
    ) -> float:
        if num_paths == 0:
            return 0.0

        reward = 0.0
        # Delivery bonus
        reward += 10.0 * delivered

        # Penalty for undelivered
        reward -= 5.0 * (num_paths - delivered)

        # Latency penalty
        reward -= 2.0 * delay

        # Energy penalty (encourage balanced usage)
        reward -= 1.0 * energy

        # Bonus for keeping nodes alive
        alive_ratio = self.swarm.num_alive / self.num_nodes
        reward += 2.0 * alive_ratio

        return reward

    def _get_obs(self) -> Dict[str, Any]:
        """Build current observation dict."""
        graph = self.graph_builder.build(self.swarm)
        node_features = self.graph_builder.get_node_feature_matrix(self.swarm, graph)
        adj = self.graph_builder.get_adjacency_matrix(graph)
        edge_attrs = self.graph_builder.get_edge_attr_matrix(graph)

        return {
            "graph": graph,
            "node_features": node_features,
            "adjacency": adj,
            "edge_attrs": edge_attrs,
            "source": self.source,
            "destinations": self.destinations,
            "alive_mask": np.array(
                [1.0 if n.is_alive else 0.0 for n in self.swarm.nodes], dtype=np.float32
            ),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, float]:
        pdr = self.packets_received / max(1, self.packets_sent)
        avg_delay = self.total_delay / max(1, self.packets_received)
        energy_used = sum(
            self._initial_energies[i] - self.swarm.nodes[i].residual_energy
            for i in range(self.num_nodes)
        )
        avg_energy = energy_used / self.num_nodes
        # Network lifetime: time until first node dies
        min_energy = min(n.residual_energy for n in self.swarm.nodes)
        return {
            "pdr": pdr,
            "avg_delay": avg_delay,
            "avg_energy_consumption": avg_energy,
            "min_residual_energy": min_energy,
            "alive_nodes": self.swarm.num_alive,
        }


# ======================================================================
# Optional NS-3 bridge (requires ns3-gym installed)
# ======================================================================

class NS3Bridge:
    """
    Thin wrapper that communicates with a running NS-3 simulation
    through the ns3-gym ZMQ interface.

    Usage:
        bridge = NS3Bridge(port=5555)
        obs = bridge.reset()
        obs, reward, done, info = bridge.step(action)

    Requires:
        - NS-3 compiled with ns3-gym module
        - The FANET simulation script running on the given port
    """

    def __init__(self, port: int = 5555):
        self.port = port
        self._env = None

    def connect(self):
        try:
            import ns3gym  # type: ignore
            from ns3gym import ns3env  # type: ignore

            self._env = ns3env.Ns3Env(port=self.port)
        except ImportError:
            raise ImportError(
                "ns3-gym is not installed. Install it from "
                "https://github.com/tkn-tub/ns3-gym or use the "
                "built-in FANETEnv for training."
            )

    def reset(self):
        assert self._env is not None, "Call connect() first"
        return self._env.reset()

    def step(self, action):
        assert self._env is not None, "Call connect() first"
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
