#!/usr/bin/env python3
"""
main.py — Entry point for the FANET AI Routing system.

Modes
-----
  train     Train the GNN + hierarchical RL routing agent.
  evaluate  Run AODV baseline comparison and generate plots.
  demo      Quick demonstration (few episodes) with visualisation.

Usage
-----
    python main.py --mode train --episodes 500
    python main.py --mode evaluate
    python main.py --mode demo
"""

import argparse
import json
import os
import sys
import random
from typing import Dict, List

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


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FANET AI-Native Multicast Routing (GNN + Deep HRL)"
    )
    p.add_argument(
        "--mode",
        choices=["train", "evaluate", "demo"],
        default="demo",
        help="Operation mode",
    )
    p.add_argument("--episodes", type=int, default=200, help="Training episodes")
    p.add_argument("--max_steps", type=int, default=100, help="Steps per episode")
    p.add_argument("--num_nodes", type=int, default=50, help="Number of UAVs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_dir", type=str, default="results", help="Output directory")
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

    print("\nDone.")


if __name__ == "__main__":
    main()
