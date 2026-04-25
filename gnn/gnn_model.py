"""
Graph Neural Network module for FANET topology embedding.

Implements a two-layer Graph Convolutional Network (GCN) that produces
per-node embeddings capturing connectivity, energy distribution,
traffic congestion, and link reliability.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Single graph convolution layer:  H' = σ(  D̃⁻¹/² Ã D̃⁻¹/² H W  )."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (N, in_features)  node feature matrix
            adj: (N, N)            adjacency matrix (binary or weighted)
        Returns:
            (N, out_features)
        """
        # Add self-loops: Ã = A + I
        identity = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        adj_hat = adj + identity

        # Degree matrix D̃
        deg = adj_hat.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Symmetric normalisation
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = adj_hat * deg_inv_sqrt * deg_inv_sqrt.t()

        # Graph convolution
        support = self.linear(x)
        out = torch.mm(adj_norm, support)
        return out


class GNNModel(nn.Module):
    """
    Two-layer GCN that produces per-node embeddings.

    Input:  (N, node_feat_dim)   – raw node features
    Output: (N, embed_dim)       – learned embeddings
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        hidden_dim: int = 32,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.conv1 = GraphConvLayer(node_feat_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, embed_dim)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, node_feat_dim)
            adjacency:     (N, N)
        Returns:
            embeddings:    (N, embed_dim)
        """
        h = F.relu(self.conv1(node_features, adjacency))
        h = self.conv2(h, adjacency)
        return h

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_numpy(
        node_features: np.ndarray,
        adjacency: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        """Convert numpy arrays to tensors on the given device."""
        device = device or torch.device("cpu")
        x = torch.from_numpy(node_features).float().to(device)
        a = torch.from_numpy(adjacency).float().to(device)
        return x, a


class GNNEncoder:
    """
    Stateful wrapper around GNNModel for easy use in the routing pipeline.

    Usage:
        encoder = GNNEncoder()
        embeddings = encoder.encode(node_features_np, adjacency_np)
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = GNNModel(node_feat_dim, hidden_dim, embed_dim).to(self.device)
        self.embed_dim = embed_dim

    def encode(
        self, node_features: np.ndarray, adjacency: np.ndarray
    ) -> np.ndarray:
        """
        Run a forward pass and return embeddings as numpy array.

        Args:
            node_features: (N, feat_dim)
            adjacency:     (N, N)
        Returns:
            embeddings:    (N, embed_dim)
        """
        self.model.eval()
        x, a = GNNModel.from_numpy(node_features, adjacency, self.device)
        with torch.no_grad():
            emb = self.model(x, a)
        return emb.cpu().numpy()

    def parameters(self):
        return self.model.parameters()

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
