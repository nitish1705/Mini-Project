"""
Meta Controller — the high-level agent in the hierarchical RL architecture.

Responsibilities:
  • Select relay / fork nodes in the multicast tree.
  • Decide *which* intermediate nodes should participate in forwarding.

The meta controller operates on a *global* view of the network
(GNN embeddings of all nodes) and outputs a probability distribution
over candidate relay nodes.

Algorithm: DQN with experience replay.
"""

import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MetaQNetwork(nn.Module):
    """Q-network for the meta controller."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MetaController:
    """
    High-level RL agent that selects relay nodes for a multicast session.

    State: concatenation of
        - source node embedding
        - aggregated destination embeddings (mean)
        - global graph summary (mean of all embeddings)

    Action: index into candidate relay node list.

    The meta controller is invoked once per routing decision to pick
    the set of relay / fork nodes before the intrinsic controller
    determines hop-by-hop paths.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        max_relays: int = 10,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.embed_dim = embed_dim
        self.max_relays = max_relays
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        # State = [source_emb | dest_mean_emb | global_mean_emb] → 3 * embed_dim
        state_dim = 3 * embed_dim
        action_dim = max_relays  # pick among top-k candidates

        self.q_net = MetaQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = MetaQNetwork(state_dim, action_dim).to(self.device)
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
        source: int,
        destinations: List[int],
    ) -> np.ndarray:
        """
        Build a fixed-size state vector from node embeddings.

        Args:
            embeddings: (N, embed_dim)
            source: source node id
            destinations: list of destination node ids
        """
        src_emb = embeddings[source]
        dest_embs = embeddings[destinations]
        dest_mean = dest_embs.mean(axis=0) if len(destinations) > 0 else np.zeros(self.embed_dim)
        global_mean = embeddings.mean(axis=0)
        state = np.concatenate([src_emb, dest_mean, global_mean])
        return state.astype(np.float32)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_relays(
        self,
        state: np.ndarray,
        candidate_ids: List[int],
        num_relays: int = 3,
    ) -> List[int]:
        """
        Select relay nodes from the candidate list.

        Uses ε-greedy exploration.
        """
        if len(candidate_ids) == 0:
            return []

        # Pad or truncate candidates to max_relays
        effective = candidate_ids[:self.max_relays]

        if random.random() < self.epsilon:
            k = min(num_relays, len(effective))
            return random.sample(effective, k)

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        # Map Q-values to candidates
        q_for_cands = q_values[:len(effective)]
        top_indices = np.argsort(q_for_cands)[::-1][:num_relays]
        return [effective[i] for i in top_indices]

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

        # Current Q
        q_current = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft-update target network
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
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
