# AI-Native Multicast Routing in FANETs

Graph Neural Networks + Deep Hierarchical Reinforcement Learning for energy-aware multicast routing in Flying Ad Hoc Networks.

This repository is a working simplified prototype of the methodology described in the report/paper: FANET multicast routing using graph-aware state encoding, hierarchical DQN controllers, loop-safe path construction, energy-aware reward design, checkpointed training, inference, and AODV-style baseline comparison.

The current review-ready goal is to present the project as an algorithmic prototype that follows the paper's core design. Full `ns-3` + `ns3-gym`, centralized critic training, true shared multicast trees, formal ablation studies, and paper-level PDR results are marked as future or extended work.

## Current Scope

This project implements the routing logic, learning pipeline, graph encoding, simulation loop, metrics, checkpoints, and evaluation plots in Python.

Implemented:

- Python FANET simulator with UAV mobility, energy consumption, communication range, packet loss, link delay, and multicast sessions.
- 50-node default FANET setup over a 2000 x 2000 m area.
- Source-to-multiple-destination multicast routing model.
- 8-dimensional UAV node feature representation.
- 4-dimensional link feature extraction.
- 2-layer GCN-style graph encoder producing 16-dimensional node embeddings.
- Meta Controller for strategic relay/fork-node selection.
- Intrinsic Controller for hop-by-hop next-hop routing.
- DQN-style learning with replay buffers.
- Energy/delay/PDR-aware reward function.
- Hop-level reward sharing for intrinsic path learning.
- Explicit visited-set loop prevention.
- Strict maximum hop limit of 15.
- Train, evaluate, demo, and infer command-line modes.
- Checkpoint saving and loading.
- AODV-style shortest-path baseline.
- Plot generation for reward, PDR, latency, energy, and controller losses.

Not fully implemented:

- Real `ns-3` execution with actual `ns3-gym` runtime traces.
- Hardware-level RF, MAC collision, and Wi-Fi PHY simulation.
- Physical UAV deployment.

## Review-Ready Position

The main algorithmic parts of the paper can be represented and tested without a full `ns-3` backend. This repository currently implements these core layers:

1. FANET state generation.
2. Dynamic graph construction.
3. GNN-based topology encoding.
4. Hierarchical routing decision logic.
5. Energy-aware reward function.
6. Loop-safe multicast path construction.
7. RL training and checkpointing.
8. Inference from saved models.
9. AODV baseline comparison.
10. Result visualization and JSON logging.

The implementation should be presented as **similar to the paper at the architecture/prototype level**, not as a complete reproduction of every experimental claim in the paper.

## Next-Hour Implementation Plan

These are the realistic improvements that can be completed quickly before review. They improve correctness and presentation without attempting large research features.

### Priority 1: Keep Documentation Honest

Status: feasible immediately.

Update all project documents to say:

```text
The project implements a Python-based GNN-HDRL FANET routing prototype.
It follows the paper's main architecture: graph state encoding, Meta and
Intrinsic DQN controllers, energy-aware reward, visited-set loop prevention,
hop limit, checkpointed training/inference, and AODV-style comparison.
Full ns-3/ns3-gym, centralized critic CTDE, formal ablation studies, and
paper-level PDR are future extensions.
```

### Priority 2: Run a Clean Demo Validation

Status: feasible within the hour.

Run:

```bash
python -m py_compile main.py simulation/ns3_env.py simulation/uav_node.py graph/graph_builder.py gnn/gnn_model.py rl/meta_controller.py rl/intrinsic_controller.py rl/training.py routing/routing_engine.py evaluation/metrics.py evaluation/plots.py
python main.py --mode demo
python main.py --mode evaluate --num_nodes 50 --max_steps 50 --save_dir results
```

Record only what the run actually produces. Do not claim `>85%` PDR unless a run actually shows it after valid destination checking.

### Priority 3: Present AODV Honestly

Status: feasible immediately.

Use the phrase:

```text
AODV-style shortest-path baseline
```

Do not call it a full AODV protocol implementation unless route request, route reply, route error, sequence number, route cache, and route repair behavior are implemented.

### Priority 4: Explain the Remaining Paper Gaps

Status: feasible immediately.

Use this wording in review:

```text
The current implementation focuses on validating the AI routing logic in a
Python simulator. The same routing decision pipeline can later be connected
to ns3-gym for full ns-3 PHY/MAC simulation.
```

### Priority 5: Simulation Config Passthrough

Status: implemented.

Simulation parameters `comm_range`, `max_speed`, and `max_hops` are passed from CLI/config into `FANETEnv` and `IntrinsicController`, so training, evaluate, and infer now use consistent runtime tuning parameters.

## Methodology Alignment

## Paper PDF Cross-Check

This README was checked against `FANET_Routing_paper.pdf`. The table below separates the paper methodology from the current repository status so the project can be presented accurately.

| Paper PDF claim                                               | README / repository status                                                                                   |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 50 UAVs in a 2000 x 2000 m area                               | Implemented as the default Python simulation setup.                                                          |
| 1 source and 5 multicast destinations                         | Implemented through random multicast session generation.                                                     |
| 8D node features                                              | Implemented.                                                                                                 |
| 4D edge features: signal, delay, loss, bandwidth              | Computed and stored as graph edge attributes.                                                                |
| 2-layer GCN producing 16D embeddings                          | Implemented with an 8 -> 32 -> 16 GCN-style encoder.                                                         |
| Meta Controller with 10,000 replay buffer                     | Implemented.                                                                                                 |
| Intrinsic Controller with 20,000 replay buffer                | Implemented.                                                                                                 |
| 15-hop limit and visited-set loop prevention                  | Implemented in path construction.                                                                            |
| Reward balances delivery, delay, energy, and alive-node ratio | Implemented.                                                                                                 |
| Hop-level reward sharing                                      | Implemented in the training loop.                                                                            |
| CLI modes: train, evaluate, demo, infer                       | Implemented.                                                                                                 |
| Checkpoints loaded automatically in infer mode                | Implemented.                                                                                                 |
| Python Gym-style simulation layer                             | Implemented.                                                                                                 |
| ns-3 / ns3-gym integrated simulation                          | Not fully active; future high-fidelity simulation work.                                                      |
| PyTorch Geometric GCN                                         | Current code uses a custom PyTorch GCN-style layer; PyTorch Geometric is optional/future.                    |
| 30-40 m/s high-speed mobility                                 | Targeted in the paper; current default simulator speed should be upgraded through curriculum/speed config.   |
| Centralized Critic / CTDE                                     | Claimed in the paper, not currently implemented as a separate critic module. Do not claim this unless added. |
| True shared multicast tree                                    | Current implementation uses per-destination multicast paths; shared-tree enforcement is an upgrade item.     |
| Full AODV protocol                                            | Current baseline is AODV-style shortest-path routing, not a complete RFC-level AODV stack.                   |
| PDR > 85%, AODV ~65%, delay <90 ms                            | Paper result claim; must be revalidated after metric fixes and 100-episode inference.                        |
| Ablation study: no energy penalty / no GNN                    | Paper result claim; should be added as reproducible scripts before claiming final ablation results.          |
| Curriculum learning                                           | Paper claim; future/training enhancement.                                                                    |

Summary: the README now matches the paper's core GNN-HDRL routing design, while clearly marking the strongest paper claims that are not yet fully implemented in the repository.

### Paper Claim: 8D Node Features

Implemented.

Each UAV node produces an 8-dimensional feature vector:

- normalized x position
- normalized y position
- normalized altitude
- normalized speed
- normalized residual energy
- normalized neighbor count
- normalized queue size
- traffic load placeholder

Relevant files:

- `simulation/uav_node.py`
- `graph/graph_builder.py`

### Paper Claim: 4D Edge Features

Implemented as link attributes.

Each active wireless link stores:

- signal strength
- propagation/link delay
- packet loss probability
- available bandwidth

Relevant file:

- `graph/graph_builder.py`

Note: the active GCN currently uses adjacency and node features. The edge attributes are available for routing, logging, and future edge-aware GNN upgrades.

### Paper Claim: 2-Layer GCN, 8 -> 32 -> 16

Implemented.

The graph encoder maps:

```text
8D node feature -> 32D hidden feature -> 16D node embedding
```

Relevant file:

- `gnn/gnn_model.py`

### Paper Claim: Hierarchical RL

Implemented as two DQN-style controllers.

Meta Controller:

- Selects strategic relay/fork-node candidates.
- Uses 10,000-transition replay buffer.
- Operates on source, destination, and global graph embedding summaries.

Intrinsic Controller:

- Builds local hop-by-hop paths.
- Uses 20,000-transition replay buffer.
- Enforces visited-set loop prevention.
- Enforces 15-hop maximum path budget.

Relevant files:

- `rl/meta_controller.py`
- `rl/intrinsic_controller.py`
- `rl/training.py`
- `routing/routing_engine.py`

### Paper Claim: Multi-Objective Reward

Implemented.

Reward formula:

```text
r(t) = 10 * D - 5 * (P - D) - 2 * delay - 1 * E_used + 2 * S
```

Where:

- `D` = delivered packets
- `P - D` = failed or undelivered forwarded packets
- `delay` = accumulated forwarding delay
- `E_used` = energy consumed
- `S` = alive nodes / total nodes

Relevant file:

- `simulation/ns3_env.py`

### Paper Claim: Hop-Level Reward Sharing

Implemented in training.

The intrinsic controller receives a fractional reward based on the number of hop-level transitions used in a route.

Relevant file:

- `rl/training.py`

### Paper Claim: Loop Prevention and 15-Hop Limit

Implemented.

The intrinsic path builder maintains a visited set and does not revisit already-used nodes in the same path. Routing stops after 15 hops.

Relevant file:

- `rl/intrinsic_controller.py`

### Paper Claim: Train / Evaluate / Demo / Infer Modes

Implemented.

Supported modes:

```bash
python main.py --mode train
python main.py --mode evaluate
python main.py --mode demo
python main.py --mode infer
```

Relevant file:

- `main.py`

### Paper Claim: AODV Baseline Comparison

Implemented as an AODV-style shortest-path baseline.

The baseline uses graph shortest paths to approximate reactive route discovery behavior for comparison.

Relevant file:

- `routing/routing_engine.py`

Note: this is not a full RFC-level AODV protocol implementation. It is a practical shortest-path baseline used for project comparison.

## Practical Review Checklist

Before review, focus on things that can be completed safely in about an hour:

1. Run compile validation.
2. Run `python main.py --mode demo`.
3. Run `python main.py --mode evaluate --num_nodes 50 --max_steps 50 --save_dir results`.
4. Keep the generated result plots ready.
5. Present only the metrics produced by current valid runs.
6. Use the phrase `AODV-style shortest-path baseline`.
7. State that real `ns-3`, centralized critic CTDE, formal ablations, shared multicast tree optimization, and paper-level PDR tuning are future work.

## Project Structure

```text
fanet_ai_routing/
├── main.py
├── requirements.txt
├── README.md
├── simulation/
│   ├── ns3_env.py
│   └── uav_node.py
├── graph/
│   └── graph_builder.py
├── gnn/
│   └── gnn_model.py
├── rl/
│   ├── meta_controller.py
│   ├── intrinsic_controller.py
│   └── training.py
├── routing/
│   └── routing_engine.py
├── evaluation/
│   ├── metrics.py
│   └── plots.py
└── results/
    ├── checkpoints/
    ├── training_history.json
    ├── inference_history.json
    └── *.png
```

## File Responsibilities

### `main.py`

Main command-line entry point.

Responsibilities:

- Parse CLI arguments.
- Start training, evaluation, demo, or inference.
- Save training and inference histories.
- Select checkpoints automatically.
- Run AODV-style baseline.
- Generate plots.

### `simulation/ns3_env.py`

Python FANET simulation environment.

Responsibilities:

- Reset the FANET scenario.
- Create random multicast sessions.
- Execute routing actions.
- Simulate packet forwarding.
- Compute reward.
- Track PDR, delay, energy, and alive nodes.
- Provide Gym-style `reset()` and `step()` interface.

Also contains an optional `NS3Bridge` placeholder for future real `ns3-gym` integration.

### `simulation/uav_node.py`

UAV node and swarm model.

Responsibilities:

- Store UAV position, velocity, altitude, energy, and queue state.
- Simulate Random Waypoint-style movement.
- Consume transmission and reception energy.
- Generate 8D node feature vectors.

### `graph/graph_builder.py`

Dynamic FANET graph construction.

Responsibilities:

- Build NetworkX topology graph.
- Add UAVs as graph nodes.
- Add wireless links as graph edges.
- Compute signal strength, delay, loss, and bandwidth.
- Generate node feature matrix.
- Generate adjacency matrix.
- Generate edge attribute dictionary.

### `gnn/gnn_model.py`

Graph encoder.

Responsibilities:

- Implement graph convolution layers.
- Convert node features and adjacency into 16D embeddings.
- Provide a `GNNEncoder` wrapper for routing and training.

### `rl/meta_controller.py`

Strategic high-level DQN controller.

Responsibilities:

- Build meta state from source, destination, and global embeddings.
- Select relay/fork-node candidates.
- Store replay transitions.
- Train Q-network.
- Save/load checkpoint.

### `rl/intrinsic_controller.py`

Local hop-by-hop DQN controller.

Responsibilities:

- Build intrinsic state from current node, destination, and neighbors.
- Select next-hop node.
- Construct path.
- Prevent loops using visited set.
- Enforce 15-hop maximum.
- Store replay transitions.
- Train Q-network.
- Save/load checkpoint.

### `rl/training.py`

Training pipeline.

Responsibilities:

- Run episodes.
- Encode graph state.
- Generate routing actions.
- Execute environment step.
- Store Meta and Intrinsic transitions.
- Apply hop-level reward sharing.
- Update controller networks.
- Save checkpoints.

### `routing/routing_engine.py`

Routing decision engine.

Responsibilities:

- Combine GNN, Meta Controller, and Intrinsic Controller.
- Produce multicast routing action.
- Build paths for each destination.
- Provide AODV-style shortest-path baseline.

### `evaluation/metrics.py`

Metric helper functions.

Tracks:

- Packet Delivery Ratio
- End-to-end delay
- Throughput helper
- Energy consumption
- Network lifetime helper

### `evaluation/plots.py`

Plot generation.

Generates:

- `reward_curve.png`
- `pdr_comparison.png`
- `latency_comparison.png`
- `energy_consumption.png`
- `loss_curves.png`

## Setup

```bash
cd "/Users/nitishm/Desktop/MINI PROJECT/Project/fanet_ai_routing"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Requirements

Core dependencies:

- Python 3.10+
- PyTorch
- NumPy
- NetworkX
- Matplotlib

Optional future dependencies:

- PyTorch Geometric
- TensorBoard
- ns-3
- ns3-gym

## Run Commands

### Quick Demo

Runs a small training/demo pass.

```bash
python main.py --mode demo
```

Tuned demo example:

```bash
python main.py --mode demo --num_nodes 50 --max_steps 50 --comm_range 250 --max_speed 15 --max_hops 15 --save_dir results
```

### Full Training

```bash
python main.py --mode train --episodes 500 --max_steps 100 --num_nodes 50 --save_dir results
```

Recommended stronger run:

```bash
python main.py --mode train --episodes 1000 --max_steps 100 --num_nodes 50 --save_dir results
```

Outputs:

- `results/checkpoints/meta_ep*.pt`
- `results/checkpoints/intrinsic_ep*.pt`
- `results/training_history.json`
- `results/reward_curve.png`
- `results/pdr_comparison.png`
- `results/latency_comparison.png`
- `results/energy_consumption.png`
- `results/loss_curves.png`

### Evaluate Existing Training History

```bash
python main.py --mode evaluate --num_nodes 50 --max_steps 100 --save_dir results
```

This reads:

```text
results/training_history.json
```

Then runs the AODV-style baseline and regenerates plots.

### Inference from Saved Checkpoints

Auto-select the best matching checkpoint pair:

```bash
python main.py --mode infer --infer_episodes 100 --max_steps 100 --num_nodes 50 --save_dir results
```

Use explicit checkpoints:

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
- refreshed plots in `results/*.png`

## CLI Arguments

```text
--mode             train, evaluate, demo, or infer
--episodes         number of training episodes
--infer_episodes   number of inference episodes
--max_steps        maximum routing decisions per episode
--num_nodes        number of UAV nodes
--comm_range       UAV communication range in meters
--max_speed        UAV speed upper bound in m/s
--max_hops         maximum intrinsic-controller hop budget
--seed             random seed
--save_dir         output directory
--meta_ckpt        explicit Meta Controller checkpoint
--intrinsic_ckpt   explicit Intrinsic Controller checkpoint
```

## Expected Output Files

```text
results/
├── training_history.json
├── inference_history.json
├── simulator_note.txt
├── reward_curve.png
├── pdr_comparison.png
├── latency_comparison.png
├── energy_consumption.png
├── loss_curves.png
└── checkpoints/
    ├── meta_ep100.pt
    ├── intrinsic_ep100.pt
    ├── meta_ep200.pt
    ├── intrinsic_ep200.pt
    └── ...
```

## Metric Definitions

### Packet Delivery Ratio

```text
PDR = packets_received / packets_sent
```

### Average End-to-End Delay

```text
avg_delay = total_delay / delivered_packets
```

### Energy Consumption

Energy is computed from:

- UAV movement energy
- transmit energy
- receive energy

### Reward

```text
r(t) = 10 * delivered
       - 5 * failed
       - 2 * delay
       - 1 * energy
       + 2 * alive_ratio
```

## Validation Procedure

Use this sequence before final project evaluation.

### 1. Compile Check

```bash
python -m py_compile \
  main.py \
  simulation/ns3_env.py \
  simulation/uav_node.py \
  graph/graph_builder.py \
  gnn/gnn_model.py \
  rl/meta_controller.py \
  rl/intrinsic_controller.py \
  rl/training.py \
  routing/routing_engine.py \
  evaluation/metrics.py \
  evaluation/plots.py
```

### 2. Smoke Test

```bash
python main.py --mode demo
```

### 3. Training Test

```bash
python main.py --mode train --episodes 100 --max_steps 50 --num_nodes 50 --save_dir results
```

### 4. Inference Test

```bash
python main.py --mode infer --infer_episodes 100 --max_steps 100 --num_nodes 50 --save_dir results
```

### 5. Plot Test

```bash
python main.py --mode evaluate --num_nodes 50 --max_steps 100 --save_dir results
```

## How to Present This Project

Recommended honest description:

```text
This project implements a Python-based FANET multicast routing prototype using
GNN-assisted hierarchical DQN controllers. It follows the core routing methodology
of the paper, including graph state encoding, Meta/Intrinsic controllers,
energy-aware rewards, loop prevention, checkpointed training, inference, and
AODV-style baseline comparison. The remaining future extension is replacing the
Python simulator with a full ns-3/ns3-gym backend for higher-fidelity PHY/MAC
validation.
```

Avoid claiming:

- Real `ns-3` simulation has already been fully executed.
- Real MAC/PHY collision modeling is complete.
- A full centralized critic is implemented, unless that module is added.
- A full RFC-level AODV protocol is implemented.
- Physical UAV deployment has been completed.

## Known Limitations

- The current simulator is Python-based and approximates network behavior.
- The optional `NS3Bridge` is present, but real `ns3-gym` integration is not active by default.
- The AODV baseline is an AODV-style shortest-path reference, not a full protocol stack.
- The current graph encoder uses node features and adjacency; edge attributes are computed but should be integrated more deeply for full edge-aware GNN behavior.
- True shared multicast-tree construction can be improved further.
- Real-world RF interference, GPS error, weather, hardware processing delay, and UAV flight dynamics are not modeled fully.

## Future Work: Real ns-3 + ns3-gym Integration

The future high-fidelity simulation upgrade is to connect the existing routing logic to a real `ns-3` simulation.

Required future steps:

1. Install and build `ns-3`.
2. Install and configure `ns3-gym`.
3. Create an `ns-3` FANET scenario with 50 UAV nodes.
4. Implement Random Waypoint or Gauss-Markov mobility in `ns-3`.
5. Configure Wi-Fi PHY/MAC.
6. Expose observations through `ns3-gym`.
7. Send routing actions from Python to `ns-3`.
8. Parse real trace metrics.
9. Compare Python-simulation results with `ns-3` results.
10. Update report results with real `ns-3` metrics.

## Recommended Final Claim

The project can be described as:

```text
A Python-based GNN-HDRL FANET multicast routing prototype with graph-aware
state representation, hierarchical DQN routing decisions, energy-aware reward
optimization, loop-safe path construction, checkpointed training/inference,
and AODV-style baseline comparison. Full ns-3/ns3-gym integration is future
high-fidelity simulation work.
```
