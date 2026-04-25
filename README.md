# FANET AI Routing (GNN + Hierarchical RL)

This project simulates UAV ad-hoc networking (FANET) and trains a hierarchical RL routing policy.

## Project Structure

- `main.py`: CLI entry point for train/evaluate/demo/infer workflows.
- `simulation/`: FANET environment and UAV mobility/energy simulation.
- `graph/`: graph construction and feature extraction.
- `gnn/`: graph encoder model.
- `rl/`: meta and intrinsic RL controllers + training loop.
- `routing/`: routing engine and AODV baselines.
- `evaluation/`: metrics and plot generation.
- `results/`: generated checkpoints, histories, and figures.

## Setup

```bash
cd "/Users/nitishm/Desktop/MINI PROJECT/Project/fanet_ai_routing"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Commands

### 1) Quick Demo

```bash
python main.py --mode demo
```

### 2) Full Training

```bash
python main.py --mode train --episodes 500 --max_steps 100 --num_nodes 50 --save_dir results
```

Outputs:

- `results/checkpoints/meta_ep*.pt`
- `results/checkpoints/intrinsic_ep*.pt`
- `results/training_history.json`
- plots in `results/*.png`

### 3) Plot Regeneration + AODV Baseline Comparison

```bash
python main.py --mode evaluate --num_nodes 50 --max_steps 100 --save_dir results
```

This reads `results/training_history.json`, runs AODV baseline episodes, and refreshes plots.

### 4) Inference from Saved Checkpoints (Policy-Only)

Auto-select best checkpoint pair from `results/checkpoints`:

```bash
python main.py --mode infer --infer_episodes 100 --max_steps 100 --num_nodes 50 --save_dir results
```

Specify explicit checkpoints:

```bash
python main.py --mode infer \
  --infer_episodes 100 \
  --max_steps 100 \
  --num_nodes 50 \
  --save_dir results \
  --meta_ckpt results/checkpoints/meta_ep500.pt \
  --intrinsic_ckpt results/checkpoints/intrinsic_ep500.pt
```

Inference outputs:

- `results/inference_history.json`
- refreshed comparison plots in `results/*.png`

## Parameters Cheat Sheet

- `--episodes`: number of training episodes.
- `--infer_episodes`: number of inference episodes in `--mode infer`.
- `--max_steps`: max routing decisions per episode.
- `--num_nodes`: number of UAVs in the FANET simulation.
- `--save_dir`: output directory for histories, checkpoints, and plots.

## Notes

- Checkpoint auto-pick in infer mode chooses the best available paired episode based on training reward from `training_history.json` when available, otherwise falls back to latest paired checkpoint.
- AODV baseline is used for comparison plots in evaluate/infer workflows.
