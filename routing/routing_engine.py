"""
Routing Decision Engine.

Integrates the GNN encoder, meta controller, and intrinsic controller
to produce a complete multicast routing action for the FANET environment.

Workflow per timestep:
    1. Encode network state with GNN  →  node embeddings.
    2. Meta controller selects relay / fork nodes.
    3. Intrinsic controller builds hop-by-hop path for each destination.
    4. Return the combined routing action.

Also includes an AODV baseline for comparison.
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from gnn.gnn_model import GNNEncoder
from rl.meta_controller import MetaController
from rl.intrinsic_controller import IntrinsicController


class RoutingEngine:
    """Multicast routing engine powered by GNN + hierarchical RL."""

    def __init__(
        self,
        gnn_encoder: GNNEncoder,
        meta_controller: MetaController,
        intrinsic_controller: IntrinsicController,
    ):
        self.gnn = gnn_encoder
        self.meta = meta_controller
        self.intrinsic = intrinsic_controller

    def compute_action(
        self,
        obs: Dict[str, Any],
        source: int,
        destinations: List[int],
    ) -> Tuple[Dict[str, Any], np.ndarray, List[Tuple]]:
        """
        Produce a routing action from the current observation.

        Returns:
            action:       dict with "relay_nodes" and "paths"
            meta_state:   state vector used by the meta controller
            all_transitions: list of intrinsic-controller transitions
        """
        node_features = obs["node_features"]
        adjacency = obs["adjacency"]
        graph: nx.Graph = obs["graph"]

        # 1. GNN embedding
        embeddings = self.gnn.encode(node_features, adjacency)

        # 2. Meta controller — select relay nodes
        meta_state = self.meta.build_state(embeddings, source, destinations)

        # Candidate relays: nodes that are not source/destination & have high degree
        candidate_relays = [
            n
            for n in graph.nodes
            if n != source and n not in destinations
        ]
        # Sort by degree (heuristic for good relay candidates)
        candidate_relays.sort(
            key=lambda n: graph.degree(n) if n in graph else 0, reverse=True
        )

        relay_nodes = self.meta.select_relays(
            meta_state, candidate_relays, num_relays=min(5, len(candidate_relays))
        )

        # 3. Intrinsic controller — build paths for each destination
        paths: Dict[int, List[int]] = {}
        all_transitions: List[Tuple] = []

        for dest in destinations:
            path, transitions = self.intrinsic.build_path(
                embeddings, source, dest, adjacency
            )
            paths[dest] = path
            all_transitions.extend(transitions)

        action = {
            "relay_nodes": relay_nodes,
            "paths": paths,
        }

        return action, meta_state, all_transitions


# ======================================================================
# AODV baseline for performance comparison
# ======================================================================

class AODVBaseline:
    """
    Simplified AODV-style shortest-path routing baseline.

    Uses NetworkX shortest path (Dijkstra with hop count) to route
    packets from source to each multicast destination independently.
    """

    def compute_action(
        self,
        obs: Dict[str, Any],
        source: int,
        destinations: List[int],
    ) -> Dict[str, Any]:
        graph: nx.Graph = obs["graph"]
        paths: Dict[int, List[int]] = {}

        for dest in destinations:
            try:
                path = nx.shortest_path(graph, source, dest)
                paths[dest] = path
            except nx.NetworkXNoPath:
                paths[dest] = [source]  # unreachable

        return {"relay_nodes": [], "paths": paths}


class WeightedAODVBaseline:
    """
    AODV variant that uses edge weights derived from link quality.

    Edge weight = delay + loss_penalty, so Dijkstra picks the
    highest-quality path rather than the minimum-hop path.
    """

    def compute_action(
        self,
        obs: Dict[str, Any],
        source: int,
        destinations: List[int],
    ) -> Dict[str, Any]:
        graph: nx.Graph = obs["graph"]

        # Build a weighted copy
        wg = nx.Graph()
        for u, v, data in graph.edges(data=True):
            weight = data.get("delay", 0.001) + data.get("loss", 0.0) * 10.0
            wg.add_edge(u, v, weight=weight)
        for n in graph.nodes:
            if n not in wg:
                wg.add_node(n)

        paths: Dict[int, List[int]] = {}
        for dest in destinations:
            try:
                path = nx.shortest_path(wg, source, dest, weight="weight")
                paths[dest] = path
            except nx.NetworkXNoPath:
                paths[dest] = [source]

        return {"relay_nodes": [], "paths": paths}
