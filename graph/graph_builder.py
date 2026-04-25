"""
Graph construction module for FANET topology.

Builds a NetworkX graph from the UAV swarm state.
Nodes carry feature vectors; edges carry link-quality attributes.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from simulation.uav_node import UAVNode, UAVSwarm


class GraphBuilder:
    """Builds and maintains the FANET topology graph."""

    def __init__(self, comm_range: float = 250.0):
        self.comm_range = comm_range

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build(self, swarm: UAVSwarm) -> nx.Graph:
        """
        Construct a NetworkX graph from the current swarm state.

        Nodes = alive UAVs with feature attributes.
        Edges = wireless links between nodes within comm range.
        """
        G = nx.Graph()
        alive_nodes = swarm.get_alive_nodes()

        # Add nodes
        for node in alive_nodes:
            G.add_node(
                node.node_id,
                position=node.position,
                velocity=node.velocity,
                energy=node.residual_energy,
                speed=node.speed,
                queue_size=node.queue_size,
            )

        # Add edges (wireless links)
        for i, n1 in enumerate(alive_nodes):
            for n2 in alive_nodes[i + 1 :]:
                dist = n1.distance_to(n2)
                if dist <= self.comm_range:
                    # Compute link quality metrics
                    signal_strength = self._signal_strength(dist)
                    delay = self._link_delay(dist)
                    loss_prob = self._packet_loss_prob(dist)
                    bandwidth = self._available_bandwidth(dist)

                    G.add_edge(
                        n1.node_id,
                        n2.node_id,
                        distance=dist,
                        signal_strength=signal_strength,
                        delay=delay,
                        loss=loss_prob,
                        bandwidth=bandwidth,
                    )

        # Fill in neighbor counts
        for nid in G.nodes:
            G.nodes[nid]["neighbor_count"] = G.degree(nid)

        return G

    # ------------------------------------------------------------------
    # Feature matrices for GNN
    # ------------------------------------------------------------------

    def get_node_feature_matrix(
        self, swarm: UAVSwarm, graph: nx.Graph
    ) -> np.ndarray:
        """
        Return an (N, 8) feature matrix aligned with node indices 0..N-1.

        Features per node:
            [x_norm, y_norm, z_norm, speed_norm, energy_norm,
             neighbor_count_norm, queue_norm, traffic_load]
        """
        num_nodes = len(swarm.nodes)
        features = np.zeros((num_nodes, 8), dtype=np.float32)

        for node in swarm.nodes:
            fv = node.feature_vector()
            # Patch in neighbor count from graph
            if node.node_id in graph:
                fv[5] = graph.degree(node.node_id) / max(1, num_nodes - 1)
            features[node.node_id] = fv

        return features

    def get_adjacency_matrix(self, graph: nx.Graph) -> np.ndarray:
        """Return dense adjacency matrix for the graph.  Shape (N, N)."""
        node_list = sorted(graph.nodes)
        A = nx.to_numpy_array(graph, nodelist=node_list, dtype=np.float32)
        return A

    def get_edge_attr_matrix(self, graph: nx.Graph) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Return a dict mapping (src, dst) -> 4-dim edge attribute vector:
            [signal_strength, delay, loss_prob, bandwidth]
        """
        edge_attrs = {}
        for u, v, data in graph.edges(data=True):
            vec = np.array(
                [
                    data.get("signal_strength", 0.0),
                    data.get("delay", 0.0),
                    data.get("loss", 0.0),
                    data.get("bandwidth", 0.0),
                ],
                dtype=np.float32,
            )
            edge_attrs[(u, v)] = vec
            edge_attrs[(v, u)] = vec
        return edge_attrs

    # ------------------------------------------------------------------
    # Link quality models
    # ------------------------------------------------------------------

    def _signal_strength(self, distance: float) -> float:
        """Free-space path loss model (simplified dBm → normalised 0-1)."""
        if distance < 1.0:
            distance = 1.0
        # RSSI ∝ 1/d²  → normalised to [0, 1]
        return min(1.0, (self.comm_range / distance) ** 2)

    def _link_delay(self, distance: float) -> float:
        """Propagation + queuing delay (seconds)."""
        propagation = distance / 3e8
        queuing = random.uniform(0, 0.002)
        return propagation + queuing

    def _packet_loss_prob(self, distance: float) -> float:
        """Simple distance-based loss probability."""
        ratio = distance / self.comm_range
        return min(0.5, ratio ** 2 * 0.3)

    def _available_bandwidth(self, distance: float) -> float:
        """Normalised available bandwidth (1.0 = full, ~0 = congested)."""
        base = 1.0 - (distance / self.comm_range) * 0.5
        jitter = random.uniform(-0.05, 0.05)
        return max(0.1, min(1.0, base + jitter))
