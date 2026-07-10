# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_projection.py
"""Sparse projection: dense embedding -> sparse 50K activation via k-winners-take-all."""

import torch
import torch.nn.functional as F
from .config import PSNConfig


class SparseProjection:
    """Projects a dense embedding vector into a sparse high-dimensional activation pattern.

    Uses a fixed random projection matrix (He-normal init) followed by k-winners-take-all
    to enforce exact sparsity. The top-k neurons by activation magnitude are kept; all
    others are zeroed.
    """

    def __init__(self, config: PSNConfig):
        self.config = config
        self.device = config.device

        # Random projection: [d_embedding, n_neurons]
        # He-normal: std = sqrt(2 / fan_in)
        std = (2.0 / config.d_embedding) ** 0.5
        self.W_proj = torch.randn(
            config.d_embedding, config.n_neurons,
            device=self.device, dtype=torch.float32
        ) * std

    def project(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a single embedding to sparse activation.

        Args:
            embedding: [d_embedding] dense vector

        Returns:
            activation: [n_neurons] sparse activation (mostly zeros)
            active_indices: [k_winners] indices of active neurons
        """
        # Linear projection
        h = embedding @ self.W_proj  # [n_neurons]

        # k-winners-take-all
        k = self.config.k_winners
        topk_vals, topk_idx = torch.topk(h, k)

        # Build sparse activation
        activation = torch.zeros(self.config.n_neurons, device=self.device)
        activation[topk_idx] = F.relu(topk_vals)  # only positive activations

        return activation, topk_idx

    def project_batch(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Project a batch of embeddings.

        Args:
            embeddings: [batch, d_embedding]

        Returns:
            activations: [batch, n_neurons]
            active_indices_list: list of [k_winners] tensors
        """
        h = embeddings @ self.W_proj  # [batch, n_neurons]
        k = self.config.k_winners

        topk_vals, topk_idx = torch.topk(h, k, dim=1)

        activations = torch.zeros(embeddings.shape[0], self.config.n_neurons, device=self.device)
        for i in range(embeddings.shape[0]):
            activations[i, topk_idx[i]] = F.relu(topk_vals[i])

        return activations, [topk_idx[i] for i in range(embeddings.shape[0])]

    def state_dict(self) -> dict:
        return {"W_proj": self.W_proj.cpu()}

    def load_state_dict(self, state: dict):
        self.W_proj = state["W_proj"].to(self.device)
