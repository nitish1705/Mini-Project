#!/usr/bin/env python3
"""
main.py — Entry point for the FANET AI Routing system.

Modes
-----
  train     Train the GNN + hierarchical RL routing agent.
  evaluate  Run AODV baseline comparison and generate plots.
  demo      Quick demonstration (few episodes) with visualisation.
    infer     Load saved checkpoints and run policy-only inference.

Usage
-----
    python main.py --mode train --episodes 500
    python main.py --mode evaluate
    python main.py --mode demo
    python main.py --mode infer --infer_episodes 100
"""

import argparse
import json
import os
import re
import sys
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sure project root is on the path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.ns3_env import FANETEnv
from graph.graph_builder import GraphBuilder
from gnn.gnn_model import GNNEncoder
from rl.meta_controller import MetaController
from rl.intrinsic_controller import IntrinsicController
from rl.training import Trainer
from routing.routing_engine import RoutingEngine, AODVBaseline
from evaluation.metrics import MetricsCollector
from evaluation.plots import generate_all_plots


# ------------------------------------------------------------------
# Seed for reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------
# AODV baseline evaluation
# ------------------------------------------------------------------

def run_aodv_baseline(config: dict) -> Dict[str, List[float]]:
    """Run the AODV baseline for the same number of episodes and collect metrics."""
    env = FANETEnv(config=config)
    baseline = AODVBaseline()
    collector = MetricsCollector()

    pdr_list, delay_list, energy_list = [], [], []
    num_episodes = config.get("num_episodes", 100)

    print("\n--- Running AODV Baseline ---")
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        total_reward = 0.0

        for _ in range(config.get("max_steps", 100)):
            action = baseline.compute_action(obs, env.source, env.destinations)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        metrics = env.get_metrics()
        collector.record(info)
        pdr_list.append(metrics["pdr"])
        delay_list.append(metrics["avg_delay"])
        energy_list.append(metrics["avg_energy_consumption"])

        if ep % 50 == 0:
            print(f"  AODV episode {ep:>4d} | PDR {metrics['pdr']:.3f}")

    summary = collector.summary()
    print(f"  AODV Summary → PDR: {summary.get('pdr', 0):.3f}, "
          f"Delay: {summary.get('avg_delay', 0):.4f}")

    return {"pdr": pdr_list, "delay": delay_list, "energy": energy_list}


def _checkpoint_episode_from_name(filename: str, prefix: str) -> Optional[int]:
    match = re.match(rf"^{prefix}_ep(\d+)\.pt$", filename)
    return int(match.group(1)) if match else None


def _select_checkpoint_pair(
    save_dir: str,
    meta_ckpt: Optional[str],
    intrinsic_ckpt: Optional[str],
) -> Tuple[str, str, int]:
    """
    Resolve which checkpoint pair to use for inference.

    Priority:
      1) Explicit --meta_ckpt and --intrinsic_ckpt.
      2) Best shared episode inferred from training history among available pairs.
      3) Latest shared episode pair.
    """
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    if meta_ckpt and intrinsic_ckpt:
        if not os.path.isfile(meta_ckpt):
            raise FileNotFoundError(f"Meta checkpoint not found: {meta_ckpt}")
        if not os.path.isfile(intrinsic_ckpt):
            raise FileNotFoundError(f"Intrinsic checkpoint not found: {intrinsic_ckpt}")

        meta_ep = _checkpoint_episode_from_name(os.path.basename(meta_ckpt), "meta")
        intr_ep = _checkpoint_episode_from_name(
            os.path.basename(intrinsic_ckpt), "intrinsic"
        )
        selected_ep = meta_ep if meta_ep is not None else (intr_ep or -1)
        return meta_ckpt, intrinsic_ckpt, selected_ep

    meta_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("meta_ep")]
    intr_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("intrinsic_ep")]

    meta_eps = {
        _checkpoint_episode_from_name(f, "meta"): os.path.join(checkpoints_dir, f)
        for f in meta_files
    }
    intr_eps = {
        _checkpoint_episode_from_name(f, "intrinsic"): os.path.join(checkpoints_dir, f)
        for f in intr_files
    }
    # Strip Nones from malformed names
    meta_eps = {k: v for k, v in meta_eps.items() if k is not None}
    intr_eps = {k: v for k, v in intr_eps.items() if k is not None}

    shared_eps = sorted(set(meta_eps.keys()) & set(intr_eps.keys()))
    if not shared_eps:
        raise FileNotFoundError(
            "No matching meta/intrinsic checkpoint pairs found in results/checkpoints"
        )

    # Auto-pick best by reward among available checkpoint episodes.
    history_path = os.path.join(save_dir, "training_history.json")
    if os.path.isfile(history_path):
        try:
            with open(history_path) as f:
                history = json.load(f)
            rewards = history.get("reward", [])
            reward_by_ep = {
                ep: rewards[ep - 1]
                for ep in shared_eps
                if 1 <= ep <= len(rewards)
            }
            if reward_by_ep:
                best_ep = max(reward_by_ep, key=reward_by_ep.get)
                return meta_eps[best_ep], intr_eps[best_ep], best_ep
        except Exception:
            # Fall through to latest if history parsing fails.
            pass

    latest_ep = max(shared_eps)
    return meta_eps[latest_ep], intr_eps[latest_ep], latest_ep


def run_saved_model_inference(
    config: dict,
    meta_ckpt: str,
    intrinsic_ckpt: str,
) -> Dict[str, List[float]]:
    """Run inference episodes using saved controller checkpoints (no learning)."""
    env = FANETEnv(config=config)

    gnn = GNNEncoder(
        node_feat_dim=8,
        hidden_dim=config.get("gnn_hidden", 32),
        embed_dim=config.get("embed_dim", 16),
    )
    meta = MetaController(
        embed_dim=config.get("embed_dim", 16),
        max_relays=config.get("max_relays", 10),
        lr=config.get("meta_lr", 1e-3),
        gamma=config.get("gamma", 0.99),
    )
    intrinsic = IntrinsicController(
        embed_dim=config.get("embed_dim", 16),
        max_neighbors=config.get("max_neighbors", 20),
        lr=config.get("intrinsic_lr", 1e-3),
        gamma=config.get("gamma", 0.99),
    )

    meta.load(meta_ckpt)
    intrinsic.load(intrinsic_ckpt)

    # Greedy policy for deterministic inference behavior.
    meta.epsilon = 0.0
    intrinsic.epsilon = 0.0

    router = RoutingEngine(
        gnn_encoder=gnn,
        meta_controller=meta,
        intrinsic_controller=intrinsic,
    )

    collector = MetricsCollector()
    reward_list, pdr_list, delay_list, energy_list = [], [], [], []

    num_episodes = config.get("num_episodes", 100)
    print("\n--- Running Saved-Model Inference ---")
    print(f"  meta checkpoint:      {meta_ckpt}")
    print(f"  intrinsic checkpoint: {intrinsic_ckpt}")

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0

        for _ in range(config.get("max_steps", 100)):
            action, _, _ = router.compute_action(
                obs,
                source=env.source,
                destinations=env.destinations,
            )
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break

        metrics = env.get_metrics()
        collector.record(info)
        reward_list.append(ep_reward)
        pdr_list.append(metrics["pdr"])
        delay_list.append(metrics["avg_delay"])
        energy_list.append(metrics["avg_energy_consumption"])

        if ep % 50 == 0:
            print(f"  Inference episode {ep:>4d} | PDR {metrics['pdr']:.3f}")

    summary = collector.summary()
    print(
        "  Inference Summary → "
        f"PDR: {summary.get('pdr', 0):.3f}, "
        f"Delay: {summary.get('avg_delay', 0):.4f}"
    )

    return {
        "reward": reward_list,
        "pdr": pdr_list,
        "delay": delay_list,
        "energy": energy_list,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FANET AI-Native Multicast Routing (GNN + Deep HRL)"
    )
    p.add_argument(
        "--mode",
        choices=["train", "evaluate", "demo", "infer"],
        default="demo",
        help="Operation mode",
    )
    p.add_argument("--episodes", type=int, default=200, help="Training episodes")
    p.add_argument(
        "--infer_episodes",
        type=int,
        default=100,
        help="Episodes for saved-model inference mode",
    )
    p.add_argument("--max_steps", type=int, default=100, help="Steps per episode")
    p.add_argument("--num_nodes", type=int, default=50, help="Number of UAVs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_dir", type=str, default="results", help="Output directory")
    p.add_argument(
        "--meta_ckpt",
        type=str,
        default=None,
        help="Path to meta checkpoint for --mode infer (optional)",
    )
    p.add_argument(
        "--intrinsic_ckpt",
        type=str,
        default=None,
        help="Path to intrinsic checkpoint for --mode infer (optional)",
    )
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    config = {
        "num_nodes": args.num_nodes,
        "max_steps": args.max_steps,
        "num_episodes": args.episodes,
        "save_dir": os.path.join(args.save_dir, "checkpoints"),
        "embed_dim": 16,
        "gnn_hidden": 32,
        "max_relays": 10,
        "max_neighbors": 20,
        "meta_lr": 1e-3,
        "intrinsic_lr": 1e-3,
        "gamma": 0.99,
        "log_interval": 10,
    }

    if args.mode == "demo":
        config["num_episodes"] = 30
        config["max_steps"] = 50
        print("Running quick demo (30 episodes, 50 steps)…\n")

    # ---- Train ----
    if args.mode in ("train", "demo"):
        trainer = Trainer(config=config)
        trainer.train()
        history = trainer.get_history()

        # Save raw history
        os.makedirs(args.save_dir, exist_ok=True)
        history_path = os.path.join(args.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
        print(f"\nTraining history saved to {history_path}")

        # ---- AODV baseline ----
        baseline_config = {
            "num_nodes": args.num_nodes,
            "max_steps": config["max_steps"],
            "num_episodes": config["num_episodes"],
        }
        baseline_history = run_aodv_baseline(baseline_config)

        # ---- Plots ----
        print("\n--- Generating Plots ---")
        generate_all_plots(
            history, baseline_history=baseline_history, save_dir=args.save_dir
        )

    elif args.mode == "evaluate":
        # Load saved history and regenerate plots
        history_path = os.path.join(args.save_dir, "training_history.json")
        if not os.path.exists(history_path):
            print(f"No training history found at {history_path}. Run --mode train first.")
            sys.exit(1)
        with open(history_path) as f:
            history = json.load(f)

        baseline_config = {
            "num_nodes": args.num_nodes,
            "max_steps": args.max_steps,
            "num_episodes": len(history.get("reward", [])) or 100,
        }
        baseline_history = run_aodv_baseline(baseline_config)

        print("\n--- Generating Plots ---")
        generate_all_plots(
            history, baseline_history=baseline_history, save_dir=args.save_dir
        )

    elif args.mode == "infer":
        infer_config = {
            "num_nodes": args.num_nodes,
            "max_steps": args.max_steps,
            "num_episodes": args.infer_episodes,
            "embed_dim": config["embed_dim"],
            "gnn_hidden": config["gnn_hidden"],
            "max_relays": config["max_relays"],
            "max_neighbors": config["max_neighbors"],
            "meta_lr": config["meta_lr"],
            "intrinsic_lr": config["intrinsic_lr"],
            "gamma": config["gamma"],
        }

        meta_ckpt, intrinsic_ckpt, selected_ep = _select_checkpoint_pair(
            save_dir=args.save_dir,
            meta_ckpt=args.meta_ckpt,
            intrinsic_ckpt=args.intrinsic_ckpt,
        )
        if selected_ep > 0:
            print(f"Using checkpoint pair from episode {selected_ep}")
        else:
            print("Using explicit checkpoint pair")

        history = run_saved_model_inference(
            config=infer_config,
            meta_ckpt=meta_ckpt,
            intrinsic_ckpt=intrinsic_ckpt,
        )

        # Save inference history
        os.makedirs(args.save_dir, exist_ok=True)
        infer_history_path = os.path.join(args.save_dir, "inference_history.json")
        with open(infer_history_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
        print(f"\nInference history saved to {infer_history_path}")

        baseline_config = {
            "num_nodes": args.num_nodes,
            "max_steps": args.max_steps,
            "num_episodes": args.infer_episodes,
        }
        baseline_history = run_aodv_baseline(baseline_config)

        print("\n--- Generating Plots ---")
        generate_all_plots(
            history, baseline_history=baseline_history, save_dir=args.save_dir
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
