"""
Training pipeline for the hierarchical RL routing system.

Orchestrates episode-based training:
    1.  Reset the FANET environment.
    2.  At each timestep, use the GNN encoder to get embeddings.
    3.  Meta controller selects relay nodes.
    4.  Intrinsic controller builds hop-by-hop paths.
    5.  Environment executes the routing action and returns reward.
    6.  Both controllers learn from experience.
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from gnn.gnn_model import GNNEncoder
from rl.meta_controller import MetaController
from rl.intrinsic_controller import IntrinsicController
from routing.routing_engine import RoutingEngine
from simulation.ns3_env import FANETEnv


class Trainer:
    """End-to-end training loop."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self.num_episodes: int = cfg.get("num_episodes", 500)
        self.max_steps: int = cfg.get("max_steps", 100)
        self.embed_dim: int = cfg.get("embed_dim", 16)
        self.save_dir: str = cfg.get("save_dir", "checkpoints")
        self.log_interval: int = cfg.get("log_interval", 10)

        os.makedirs(self.save_dir, exist_ok=True)

        # Environment
        env_cfg = {
            "num_nodes": cfg.get("num_nodes", 50),
            "max_steps": self.max_steps,
        }
        self.env = FANETEnv(config=env_cfg)

        # GNN Encoder
        self.gnn = GNNEncoder(
            node_feat_dim=8,
            hidden_dim=cfg.get("gnn_hidden", 32),
            embed_dim=self.embed_dim,
        )

        # Meta controller
        self.meta = MetaController(
            embed_dim=self.embed_dim,
            max_relays=cfg.get("max_relays", 10),
            lr=cfg.get("meta_lr", 1e-3),
            gamma=cfg.get("gamma", 0.99),
        )

        # Intrinsic controller
        self.intrinsic = IntrinsicController(
            embed_dim=self.embed_dim,
            max_neighbors=cfg.get("max_neighbors", 20),
            lr=cfg.get("intrinsic_lr", 1e-3),
            gamma=cfg.get("gamma", 0.99),
        )

        # Routing engine (ties everything together)
        self.router = RoutingEngine(
            gnn_encoder=self.gnn,
            meta_controller=self.meta,
            intrinsic_controller=self.intrinsic,
        )

        # Logging
        self.reward_history: List[float] = []
        self.pdr_history: List[float] = []
        self.delay_history: List[float] = []
        self.energy_history: List[float] = []
        self.meta_loss_history: List[float] = []
        self.intrinsic_loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print("=" * 60)
        print("FANET AI Routing — Hierarchical RL Training")
        print("=" * 60)

        for episode in range(1, self.num_episodes + 1):
            ep_reward, ep_info = self._run_episode()

            self.reward_history.append(ep_reward)
            self.pdr_history.append(ep_info.get("pdr", 0.0))
            self.delay_history.append(ep_info.get("avg_delay", 0.0))
            self.energy_history.append(ep_info.get("energy_used", 0.0))

            if episode % self.log_interval == 0:
                avg_r = np.mean(self.reward_history[-self.log_interval:])
                avg_pdr = np.mean(self.pdr_history[-self.log_interval:])
                print(
                    f"Episode {episode:>4d} | "
                    f"Reward {avg_r:>8.2f} | "
                    f"PDR {avg_pdr:.3f} | "
                    f"ε_meta {self.meta.epsilon:.3f} | "
                    f"ε_intr {self.intrinsic.epsilon:.3f}"
                )

            # Periodic checkpoint
            if episode % 100 == 0:
                self._save_checkpoint(episode)

        # Final save
        self._save_checkpoint(self.num_episodes)
        print("\nTraining complete.")

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(self) -> tuple:
        obs = self.env.reset()
        total_reward = 0.0
        info: Dict[str, Any] = {}

        for step in range(self.max_steps):
            # Build routing action via the engine
            action, meta_state, intrinsic_transitions = (
                self.router.compute_action(
                    obs,
                    source=self.env.source,
                    destinations=self.env.destinations,
                )
            )

            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward

            # --- Store experiences and learn ---

            # Meta controller: single transition per step
            next_embeddings = self.gnn.encode(
                next_obs["node_features"], next_obs["adjacency"]
            )
            meta_next_state = self.meta.build_state(
                next_embeddings, self.env.source, self.env.destinations
            )
            # Use action index 0 as a simplification (relay selection is multi-select)
            self.meta.store_transition(meta_state, 0, reward, meta_next_state, done)
            meta_loss = self.meta.learn()
            if meta_loss:
                self.meta_loss_history.append(meta_loss)

            # Intrinsic controller: per-hop transitions
            num_hops = len(intrinsic_transitions)
            hop_reward = reward / max(1, num_hops)
            for i, (s, a_idx, cur, nxt) in enumerate(intrinsic_transitions):
                is_last = i == num_hops - 1
                if is_last:
                    # Build terminal next state
                    neighbors_nxt = [
                        j
                        for j in range(next_obs["adjacency"].shape[0])
                        if next_obs["adjacency"][nxt, j] > 0
                    ]
                    ns = self.intrinsic.build_state(
                        next_embeddings,
                        nxt,
                        self.env.destinations[0] if self.env.destinations else 0,
                        neighbors_nxt,
                    )
                    self.intrinsic.store_transition(s, a_idx, hop_reward, ns, done)
                else:
                    next_s, next_a, _, _ = intrinsic_transitions[i + 1]
                    self.intrinsic.store_transition(s, a_idx, hop_reward, next_s, False)

            intr_loss = self.intrinsic.learn()
            if intr_loss:
                self.intrinsic_loss_history.append(intr_loss)

            obs = next_obs
            if done:
                break

        return total_reward, info

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, episode: int):
        self.meta.save(os.path.join(self.save_dir, f"meta_ep{episode}.pt"))
        self.intrinsic.save(os.path.join(self.save_dir, f"intrinsic_ep{episode}.pt"))
        print(f"  [checkpoint saved @ episode {episode}]")

    def get_history(self) -> Dict[str, List[float]]:
        return {
            "reward": self.reward_history,
            "pdr": self.pdr_history,
            "delay": self.delay_history,
            "energy": self.energy_history,
            "meta_loss": self.meta_loss_history,
            "intrinsic_loss": self.intrinsic_loss_history,
        }
