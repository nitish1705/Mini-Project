"""
Intrinsic Controller — the low-level agent in the hierarchical RL architecture.

Responsibilities:
  • Given a source, destination, and set of relay nodes, select
    the next-hop node at each forwarding step.
  • Construct the full hop-by-hop path for a single (source → destination) flow.

Algorithm: DQN with experience replay (same backbone as meta controller,
but different state/action semantics).
"""

import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class IntrinsicQNetwork(nn.Module):
    """Q-network for the intrinsic (low-level) controller."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class IntrinsicController:
    """
    Low-level RL agent that selects next-hop nodes for packet forwarding.

    State: concatenation of
        - current node embedding
        - destination node embedding
        - link features to each neighbour (aggregated)

    Action: index into the current node's neighbour list.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        max_neighbors: int = 20,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 20000,
        batch_size: int = 64,
        max_hops: int = 15,
        device: Optional[torch.device] = None,
    ):
        self.embed_dim = embed_dim
        self.max_neighbors = max_neighbors
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.max_hops = max_hops
        self.device = device or torch.device("cpu")

        # State = [current_emb | dest_emb | neighbor_summary] → 3 * embed_dim
        state_dim = 3 * embed_dim
        action_dim = max_neighbors

        self.q_net = IntrinsicQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = IntrinsicQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.update_count = 0

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def build_state(
        self,
        embeddings: np.ndarray,
        current_node: int,
        destination: int,
        neighbor_ids: List[int],
    ) -> np.ndarray:
        cur_emb = embeddings[current_node]
        dest_emb = embeddings[destination]

        if len(neighbor_ids) > 0:
            neigh_embs = embeddings[neighbor_ids]
            neigh_mean = neigh_embs.mean(axis=0)
        else:
            neigh_mean = np.zeros(self.embed_dim, dtype=np.float32)

        state = np.concatenate([cur_emb, dest_emb, neigh_mean])
        return state.astype(np.float32)

    # ------------------------------------------------------------------
    # Next-hop selection
    # ------------------------------------------------------------------

    def select_next_hop(
        self,
        state: np.ndarray,
        neighbor_ids: List[int],
    ) -> Optional[int]:
        """Select the next hop from the neighbour list using ε-greedy."""
        if len(neighbor_ids) == 0:
            return None

        effective = neighbor_ids[:self.max_neighbors]

        if random.random() < self.epsilon:
            return random.choice(effective)

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        q_for_neighbors = q_values[:len(effective)]
        best_idx = int(np.argmax(q_for_neighbors))
        return effective[best_idx]

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def build_path(
        self,
        embeddings: np.ndarray,
        source: int,
        destination: int,
        adjacency: np.ndarray,
    ) -> Tuple[List[int], List[Tuple]]:
        """
        Greedily construct a hop-by-hop path from source to destination.

        Returns:
            path: list of node ids
            transitions: list of (state, action_idx, ...) for learning
        """
        path = [source]
        visited: Set[int] = {source}
        transitions: List[Tuple] = []
        current = source

        for _ in range(self.max_hops):
            if current == destination:
                break

            # Neighbours of current node
            neighbors = [
                j
                for j in range(adjacency.shape[0])
                if adjacency[current, j] > 0 and j not in visited
            ]

            if len(neighbors) == 0:
                break

            state = self.build_state(embeddings, current, destination, neighbors)
            next_hop = self.select_next_hop(state, neighbors)

            if next_hop is None:
                break

            # Record action index relative to neighbour list
            action_idx = neighbors.index(next_hop) if next_hop in neighbors else 0

            path.append(next_hop)
            visited.add(next_hop)
            transitions.append((state, action_idx, current, next_hop))
            current = next_hop

        return path, transitions

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(list(self.buffer), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.from_numpy(np.array(states)).float().to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_current = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
