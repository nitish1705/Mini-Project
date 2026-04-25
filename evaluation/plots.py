"""
Visualization module for FANET routing experiments.

Generates publication-quality plots:
  • Training reward curve
  • Packet Delivery Ratio comparison
  • Latency comparison
  • Energy consumption
  • Network lifetime
  • Loss curves
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def _smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Simple moving average for noisy curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_reward(
    reward_history: List[float],
    save_path: str = "results/reward_curve.png",
    title: str = "Training Reward Curve",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reward_history, alpha=0.3, label="Raw")
    if len(reward_history) > 10:
        ax.plot(
            range(9, len(reward_history)),
            _smooth(reward_history),
            linewidth=2,
            label="Smoothed",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pdr_comparison(
    results: Dict[str, List[float]],
    save_path: str = "results/pdr_comparison.png",
):
    """Bar / line chart comparing PDR across methods."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, values in results.items():
        ax.plot(values, label=method, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_title("PDR Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_latency_comparison(
    results: Dict[str, List[float]],
    save_path: str = "results/latency_comparison.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, values in results.items():
        ax.plot(values, label=method, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average End-to-End Delay (s)")
    ax.set_title("Latency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_energy_consumption(
    results: Dict[str, List[float]],
    save_path: str = "results/energy_consumption.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, values in results.items():
        ax.plot(values, label=method, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Energy Consumed")
    ax.set_title("Energy Consumption")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_network_lifetime(
    results: Dict[str, float],
    save_path: str = "results/network_lifetime.png",
):
    """Bar chart of network lifetime per method."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(results.keys())
    lifetimes = [results[m] for m in methods]
    bars = ax.bar(methods, lifetimes, color=["#2196F3", "#FF9800", "#4CAF50", "#F44336"])
    ax.set_ylabel("Network Lifetime (s)")
    ax.set_title("Network Lifetime Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, lifetimes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_loss_curves(
    meta_losses: List[float],
    intrinsic_losses: List[float],
    save_path: str = "results/loss_curves.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if meta_losses:
        ax1.plot(_smooth(meta_losses, 50), linewidth=1.5)
    ax1.set_title("Meta Controller Loss")
    ax1.set_xlabel("Update Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    if intrinsic_losses:
        ax2.plot(_smooth(intrinsic_losses, 50), linewidth=1.5, color="tab:orange")
    ax2.set_title("Intrinsic Controller Loss")
    ax2.set_xlabel("Update Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_all_plots(
    history: Dict[str, List[float]],
    baseline_history: Optional[Dict[str, List[float]]] = None,
    save_dir: str = "results",
):
    """Generate the full suite of evaluation plots."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Reward curve
    plot_training_reward(
        history["reward"],
        save_path=os.path.join(save_dir, "reward_curve.png"),
    )

    # 2. PDR comparison
    pdr_data = {"GNN+HRL": history["pdr"]}
    if baseline_history and "pdr" in baseline_history:
        pdr_data["AODV"] = baseline_history["pdr"]
    plot_pdr_comparison(pdr_data, save_path=os.path.join(save_dir, "pdr_comparison.png"))

    # 3. Latency comparison
    delay_data = {"GNN+HRL": history["delay"]}
    if baseline_history and "delay" in baseline_history:
        delay_data["AODV"] = baseline_history["delay"]
    plot_latency_comparison(
        delay_data, save_path=os.path.join(save_dir, "latency_comparison.png")
    )

    # 4. Energy consumption
    energy_data = {"GNN+HRL": history["energy"]}
    if baseline_history and "energy" in baseline_history:
        energy_data["AODV"] = baseline_history["energy"]
    plot_energy_consumption(
        energy_data, save_path=os.path.join(save_dir, "energy_consumption.png")
    )

    # 5. Loss curves
    plot_loss_curves(
        history.get("meta_loss", []),
        history.get("intrinsic_loss", []),
        save_path=os.path.join(save_dir, "loss_curves.png"),
    )

    print(f"\nAll plots saved to {save_dir}/")
