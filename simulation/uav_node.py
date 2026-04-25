"""
UAV Node model for FANET simulation.

Each UAV maintains telemetry: position, velocity, residual energy,
neighbor list, signal strength, and queue size.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class UAVNode:
    """Represents a single UAV in the FANET."""

    node_id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # altitude
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    residual_energy: float = 100.0  # percentage
    queue_size: int = 0
    max_queue: int = 50
    comm_range: float = 250.0

    # Random‑waypoint state
    _target_x: float = field(default=0.0, repr=False)
    _target_y: float = field(default=0.0, repr=False)
    _wait_time: float = field(default=0.0, repr=False)

    # Simulation boundaries
    area_width: float = field(default=2000.0, repr=False)
    area_height: float = field(default=2000.0, repr=False)
    max_speed: float = field(default=15.0, repr=False)

    # Energy model parameters (Joules per second)
    _energy_fly_rate: float = field(default=0.05, repr=False)
    _energy_tx_cost: float = field(default=0.01, repr=False)
    _energy_rx_cost: float = field(default=0.005, repr=False)

    def __post_init__(self):
        if self.x == 0.0 and self.y == 0.0:
            self.x = random.uniform(0, self.area_width)
            self.y = random.uniform(0, self.area_height)
            self.z = random.uniform(50, 150)  # altitude 50-150 m
        self._pick_new_waypoint()

    # ------------------------------------------------------------------
    # Mobility: Random Waypoint Model
    # ------------------------------------------------------------------

    def _pick_new_waypoint(self):
        self._target_x = random.uniform(0, self.area_width)
        self._target_y = random.uniform(0, self.area_height)
        speed = random.uniform(1.0, self.max_speed)
        dx = self._target_x - self.x
        dy = self._target_y - self.y
        dist = math.hypot(dx, dy) or 1e-6
        self.vx = (dx / dist) * speed
        self.vy = (dy / dist) * speed
        self.vz = 0.0
        self._wait_time = 0.0

    def _reached_waypoint(self) -> bool:
        return math.hypot(self._target_x - self.x, self._target_y - self.y) < 5.0

    def step(self, dt: float = 1.0):
        """Advance the UAV state by *dt* seconds."""
        if self._wait_time > 0:
            self._wait_time -= dt
            if self._wait_time <= 0:
                self._pick_new_waypoint()
            return

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Clamp to area
        self.x = max(0, min(self.x, self.area_width))
        self.y = max(0, min(self.y, self.area_height))
        self.z = max(10, min(self.z, 200))

        # Energy drain for flying
        self.residual_energy -= self._energy_fly_rate * dt
        self.residual_energy = max(0.0, self.residual_energy)

        if self._reached_waypoint():
            self._wait_time = random.uniform(0, 2.0)

    # ------------------------------------------------------------------
    # Energy helpers
    # ------------------------------------------------------------------

    def consume_tx_energy(self, num_packets: int = 1):
        self.residual_energy -= self._energy_tx_cost * num_packets
        self.residual_energy = max(0.0, self.residual_energy)

    def consume_rx_energy(self, num_packets: int = 1):
        self.residual_energy -= self._energy_rx_cost * num_packets
        self.residual_energy = max(0.0, self.residual_energy)

    @property
    def is_alive(self) -> bool:
        return self.residual_energy > 0.0

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)

    def distance_to(self, other: "UAVNode") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def in_range(self, other: "UAVNode") -> bool:
        return self.distance_to(other) <= self.comm_range

    def feature_vector(self) -> np.ndarray:
        """Return an 8-dim feature vector for GNN input."""
        return np.array(
            [
                self.x / self.area_width,
                self.y / self.area_height,
                self.z / 200.0,
                self.speed / self.max_speed,
                self.residual_energy / 100.0,
                0.0,  # placeholder: neighbor_count (filled by graph builder)
                self.queue_size / self.max_queue,
                0.0,  # placeholder: traffic_load (filled externally)
            ],
            dtype=np.float32,
        )


class UAVSwarm:
    """Collection of UAV nodes forming the FANET."""

    def __init__(
        self,
        num_nodes: int = 50,
        area_width: float = 2000.0,
        area_height: float = 2000.0,
        comm_range: float = 250.0,
        max_speed: float = 15.0,
    ):
        self.nodes: List[UAVNode] = []
        for i in range(num_nodes):
            node = UAVNode(
                node_id=i,
                area_width=area_width,
                area_height=area_height,
                comm_range=comm_range,
                max_speed=max_speed,
            )
            self.nodes.append(node)

    def step(self, dt: float = 1.0):
        """Advance all nodes by one timestep."""
        for node in self.nodes:
            if node.is_alive:
                node.step(dt)

    def get_alive_nodes(self) -> List[UAVNode]:
        return [n for n in self.nodes if n.is_alive]

    def get_node(self, node_id: int) -> UAVNode:
        return self.nodes[node_id]

    @property
    def num_alive(self) -> int:
        return sum(1 for n in self.nodes if n.is_alive)

    def reset(self):
        """Reinitialise nodes for a new episode."""
        for node in self.nodes:
            node.x = random.uniform(0, node.area_width)
            node.y = random.uniform(0, node.area_height)
            node.z = random.uniform(50, 150)
            node.residual_energy = 100.0
            node.queue_size = 0
            node._pick_new_waypoint()
