"""
sample.py — Representative code samples from FANET AI-Native Routing project
for inclusion in research reports and presentations.

This file demonstrates key components and usage patterns of the GNN-HDRL system.
"""

import sys
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np

# =====================================================================
# 1. GNN ENCODER (Graph Neural Network for state representation)
# =====================================================================

class GNNEncoder(nn.Module):
    """
    Graph Convolutional Network (GCN) encoder for FANET topology representation.
    
    Transforms 8-dimensional node features (location, velocity, energy, traffic, queue)
    and 4-dimensional edge features (signal strength, delay, loss, bandwidth) into a
    16-dimensional embedding that captures local two-hop network structure.
    """
    
    def __init__(self, node_feat_dim: int = 8, hidden_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # Two-layer GCN: node_feat_dim -> hidden_dim -> embed_dim
        self.gcn1 = GCNConv(node_feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        """
        Forward pass: aggregate node features over neighborhood.
        
        Args:
            x: Node feature matrix [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, embed_dim]
        """
        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        x = self.gcn2(x, edge_index)  # [num_nodes, embed_dim]
        return x


# =====================================================================
# 2. HIERARCHICAL CONTROLLERS (Meta + Intrinsic DQN)
# =====================================================================

class QNetwork(nn.Module):
    """Deep Q-Network for hierarchical decision making."""
    
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class MetaController(nn.Module):
    """
    Meta Controller (Coarse Planning Layer).
    
    Responsible for strategic relay placement decisions using aggregate 
    network information via GNN embeddings. Operates at longer time scales
    with a replay buffer of 10,000 transitions.
    """
    
    def __init__(self, embed_dim: int = 16, max_relays: int = 10, 
                 lr: float = 1e-3, gamma: float = 0.99):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_relays = max_relays
        self.gamma = gamma
        self.lr = lr
        
        # Q-network: concatenated GNN embedding + neighbor features + packet info
        input_dim = 3 * embed_dim  # 48 dimensions (node + neighbor + packet)
        self.q_network = QNetwork(input_dim, hidden_dim=128, action_dim=max_relays)
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = []
        self.max_buffer_size = 10000
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def compute_action(self, state: torch.Tensor, training: bool = True) -> int:
        """
        Epsilon-greedy action selection: strategic relay zone selection.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.max_relays)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax(dim=-1).item()
    
    def update(self, batch_size: int = 32):
        """Update Q-network using experience replay."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample random batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        
        # Compute Q-targets
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.q_network(next_states).max(dim=1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # MSE loss
        loss = nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)


class IntrinsicController(nn.Module):
    """
    Intrinsic Controller (Local Execution Layer).
    
    Oversees hop-by-hop path execution with loop prevention (visited sets)
    and hop-count enforcement (max 15 hops). Operates at higher frequency
    with a replay buffer of 20,000 transitions.
    """
    
    def __init__(self, embed_dim: int = 16, max_neighbors: int = 20,
                 lr: float = 1e-3, gamma: float = 0.99):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_neighbors = max_neighbors
        self.gamma = gamma
        self.lr = lr
        
        input_dim = 3 * embed_dim
        self.q_network = QNetwork(input_dim, hidden_dim=128, action_dim=max_neighbors)
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = []
        self.max_buffer_size = 20000
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def compute_action(self, state: torch.Tensor, visited_set: set,
                       training: bool = True) -> int:
        """
        Select next hop neighbor with loop prevention via visited set.
        """
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.max_neighbors)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax(dim=-1).item()
        
        # Loop prevention: ensure selected neighbor not in visited set
        if action in visited_set:
            available_actions = [a for a in range(self.max_neighbors) if a not in visited_set]
            if available_actions:
                action = np.random.choice(available_actions)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store hop-level transition."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)


# =====================================================================
# 3. REWARD FUNCTION (Multi-objective optimization)
# =====================================================================

def compute_reward(packets_delivered: int, packets_forwarded: int,
                   total_delay: float, energy_used: float,
                   alive_nodes: int, total_nodes: int) -> float:
    """
    Multi-objective reward function balancing:
      - Packet delivery (D)
      - Path length / unsuccessful forwards (P - D)
      - End-to-end delay
      - Energy consumption
      - Network lifetime (S = alive_nodes / total_nodes)
    
    Formula:
        r(t) = 10×D - 5×(P-D) - 2×delay - 1×E_used + 2×S
    
    Args:
        packets_delivered: Count of packets reaching destinations
        packets_forwarded: Total forwarded packets
        total_delay: Sum of UAV-to-UAV delays
        energy_used: Energy consumed in current step
        alive_nodes: Number of alive UAVs
        total_nodes: Total UAVs in network
    
    Returns:
        Reward scalar
    """
    D = packets_delivered
    P = packets_forwarded
    S = alive_nodes / total_nodes if total_nodes > 0 else 0
    
    reward = (10.0 * D - 5.0 * (P - D) - 2.0 * total_delay
              - 1.0 * energy_used + 2.0 * S)
    return reward


def compute_hop_level_reward(base_reward: float, hop_count: int) -> float:
    """
    Distribute base reward across hops to solve credit assignment problem.
    Each intermediate forwarder receives r_i(t) = r(t) / h, where h = hop count.
    """
    return base_reward / hop_count if hop_count > 0 else 0


# =====================================================================
# 4. TRAINING LOOP (Main learning algorithm)
# =====================================================================

def train_episode(env, gnn_encoder, meta_controller, intrinsic_controller,
                  max_steps: int = 100, max_hops: int = 15) -> float:
    """
    Execute one training episode: collect transitions, compute rewards,
    update both controllers.
    
    Args:
        env: FANET simulation environment
        gnn_encoder: GNN for state representation
        meta_controller: Strategic layer
        intrinsic_controller: Tactical layer
        max_steps: Maximum steps per episode
        max_hops: Maximum allowed hops (15)
    
    Returns:
        Total episode reward
    """
    obs = env.reset()
    total_reward = 0.0
    
    for step in range(max_steps):
        # Extract node features and build graph
        node_features = torch.tensor(obs["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
        
        # GNN forward: compute node embeddings
        embeddings = gnn_encoder(node_features, edge_index)
        
        # Meta controller: choose relay zone
        meta_state = torch.cat([embeddings[0].unsqueeze(0)] * 3, dim=1)  # Simplified state
        relay_action = meta_controller.compute_action(meta_state, training=True)
        
        # Intrinsic controller: select next hop with loop prevention
        visited = set()
        intrinsic_state = torch.cat([embeddings[1].unsqueeze(0)] * 3, dim=1)
        hop_action = intrinsic_controller.compute_action(intrinsic_state, visited, training=True)
        
        # Execute action in environment
        obs, step_reward, done, info = env.step({"meta": relay_action, "intrinsic": hop_action})
        
        # Store transitions with experience replay
        meta_controller.remember(meta_state, relay_action, step_reward, meta_state, done)
        intrinsic_controller.remember(intrinsic_state, hop_action, 
                                     compute_hop_level_reward(step_reward, info.get("hop_count", 1)),
                                     intrinsic_state, done)
        
        total_reward += step_reward
        if done:
            break
    
    # Update both controllers
    meta_controller.update(batch_size=32)
    intrinsic_controller.update(batch_size=32)
    
    return total_reward


# =====================================================================
# 5. LOOP PREVENTION & ROUTING LOGIC
# =====================================================================

class RoutingEngine:
    """
    Main routing engine combining GNN embeddings and hierarchical controllers
    with explicit loop prevention and hop-count enforcement.
    """
    
    def __init__(self, gnn_encoder, meta_controller, intrinsic_controller):
        self.gnn = gnn_encoder
        self.meta = meta_controller
        self.intrinsic = intrinsic_controller
        self.max_hops = 15
    
    def compute_multicast_path(self, source: int, destination: int,
                                network_graph, node_embeddings) -> list:
        """
        Compute multicast path from source to destination with loop prevention.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            network_graph: Current network topology
            node_embeddings: GNN embeddings for all nodes
        
        Returns:
            Path as list of node IDs, or empty list if no valid path
        """
        path = [source]
        visited = {source}
        current = source
        hop_count = 0
        
        while current != destination and hop_count < self.max_hops:
            # Get neighbors of current node
            neighbors = list(network_graph.neighbors(current))
            
            # Filter out visited nodes (loop prevention)
            available_neighbors = [n for n in neighbors if n not in visited]
            
            if not available_neighbors:
                # Dead end - no unvisited neighbors
                return []
            
            # Use intrinsic controller to select next hop
            neighbor_embeddings = torch.stack([node_embeddings[n] for n in available_neighbors])
            current_embedding = node_embeddings[current].unsqueeze(0)
            state = torch.cat([current_embedding] * 3, dim=1)
            
            next_hop_idx = self.intrinsic.compute_action(state, visited, training=False)
            next_hop = available_neighbors[next_hop_idx % len(available_neighbors)]
            
            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            hop_count += 1
        
        return path if current == destination else []


# =====================================================================
# 6. USAGE EXAMPLE
# =====================================================================

def example_training():
    """
    Example showing how to initialize and train the GNN-HDRL system.
    """
    # Hyperparameters
    config = {
        "num_nodes": 50,
        "embed_dim": 16,
        "gnn_hidden": 32,
        "max_relays": 10,
        "max_neighbors": 20,
        "meta_lr": 1e-3,
        "intrinsic_lr": 1e-3,
        "gamma": 0.99,
        "num_episodes": 500,
        "max_steps": 100,
    }
    
    # Initialize components
    gnn = GNNEncoder(node_feat_dim=8, hidden_dim=config["gnn_hidden"],
                     embed_dim=config["embed_dim"])
    
    meta = MetaController(embed_dim=config["embed_dim"],
                          max_relays=config["max_relays"],
                          lr=config["meta_lr"],
                          gamma=config["gamma"])
    
    intrinsic = IntrinsicController(embed_dim=config["embed_dim"],
                                    max_neighbors=config["max_neighbors"],
                                    lr=config["intrinsic_lr"],
                                    gamma=config["gamma"])
    
    # Training loop (pseudo-code; requires actual env)
    """
    for episode in range(config["num_episodes"]):
        reward = train_episode(env, gnn, meta, intrinsic,
                              max_steps=config["max_steps"])
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Reward = {reward:.2f}")
    """


# =====================================================================
# 7. PERFORMANCE METRICS
# =====================================================================

class MetricsCollector:
    """Collects and aggregates routing performance metrics."""
    
    def __init__(self):
        self.pdr_list = []      # Packet Delivery Ratio
        self.delay_list = []    # End-to-end delay
        self.energy_list = []   # Energy consumption
    
    def record(self, pdr: float, delay: float, energy: float):
        self.pdr_list.append(pdr)
        self.delay_list.append(delay)
        self.energy_list.append(energy)
    
    def summary(self):
        return {
            "pdr": np.mean(self.pdr_list) if self.pdr_list else 0.0,
            "delay": np.mean(self.delay_list) if self.delay_list else 0.0,
            "energy": np.mean(self.energy_list) if self.energy_list else 0.0,
        }


if __name__ == "__main__":
    print("Sample code for FANET AI-Native Multicast Routing (GNN + HDRL)")
    print("\nKey components:")
    print("  1. GNN Encoder: 2-layer GCN converting 8D node & 4D edge features to 16D embeddings")
    print("  2. Meta Controller: Strategic relay placement (10K replay buffer)")
    print("  3. Intrinsic Controller: Local hop selection (20K replay buffer)")
    print("  4. Reward: Multi-objective function balancing PDR, delay, energy, lifetime")
    print("  5. Loop Prevention: Explicit visited sets per packet flow")
    print("  6. Hop Limit: Maximum 15 hops enforced")
    print("\nFor full implementation, see main.py, gnn/gnn_model.py, rl/training.py")
