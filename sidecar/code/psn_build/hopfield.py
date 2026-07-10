# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_hopfield.py
"""Hopfield Network: sparse block architecture with attractor dynamics."""

import torch
import torch.nn.functional as F
from .config import PSNConfig


class HopfieldNetwork:
    """Modern Hopfield Network with sparse block connectivity.

    Architecture:
        - 100 blocks of 500 neurons each (50K total)
        - Dense symmetric connectivity within each block (W_intra)
        - Sparse connectivity between blocks (W_inter)
        - Energy-based attractor dynamics for pattern retrieval

    The energy function: E(xi) = -0.5 * xi^T @ W @ xi
    where W is the combined block-structured weight matrix.
    Stored patterns are energy minima; retrieval is gradient descent on E.
    """

    def __init__(self, config: PSNConfig):
        self.config = config
        self.device = config.device

        # Intra-block weights: [n_blocks, block_size, block_size]
        # Symmetric, zero diagonal - initialized to zero (no stored patterns yet)
        self.W_intra = torch.zeros(
            config.n_blocks, config.block_size, config.block_size,
            device=self.device, dtype=torch.float32
        )

        # Inter-block connectivity structure
        # Each block connects to K nearest blocks (wrap-around for topology)
        self.block_adjacency = self._build_block_adjacency()

        # Inter-block weights: sparse COO tensor [n_neurons, n_neurons]
        self._inter_indices, self._inter_values = self._init_inter_block_weights()

    def _build_block_adjacency(self) -> dict[int, list[int]]:
        """Build block adjacency: each block connects to K nearest neighbors (circular)."""
        adj = {}
        n = self.config.n_blocks
        k = self.config.inter_block_k
        for b in range(n):
            neighbors = []
            for offset in range(1, k // 2 + 1):
                neighbors.append((b + offset) % n)
                neighbors.append((b - offset) % n)
            # If k is odd, add one more forward neighbor
            if k % 2 == 1:
                neighbors.append((b + k // 2 + 1) % n)
            adj[b] = sorted(set(neighbors))[:k]
        return adj

    def _init_inter_block_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize sparse inter-block connectivity.

        For each connected block pair (b1, b2), create random connections
        at the specified density. Weights start at zero (no stored patterns).
        """
        row_indices = []
        col_indices = []

        bs = self.config.block_size
        density = self.config.inter_block_density

        for b1, neighbors in self.block_adjacency.items():
            for b2 in neighbors:
                if b2 <= b1:
                    continue  # symmetric: only store upper triangle
                # Random connections at specified density
                n_connections = int(bs * bs * density)
                src_neurons = torch.randint(0, bs, (n_connections,))
                dst_neurons = torch.randint(0, bs, (n_connections,))

                # Global indices
                global_src = b1 * bs + src_neurons
                global_dst = b2 * bs + dst_neurons

                # Both directions (symmetric)
                row_indices.extend(global_src.tolist())
                col_indices.extend(global_dst.tolist())
                row_indices.extend(global_dst.tolist())
                col_indices.extend(global_src.tolist())

        if len(row_indices) == 0:
            indices = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            values = torch.zeros(0, device=self.device)
        else:
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=self.device)
            values = torch.zeros(len(row_indices), device=self.device)

        return indices, values

    def _get_inter_sparse(self) -> torch.Tensor:
        """Build sparse COO tensor from current indices and values."""
        return torch.sparse_coo_tensor(
            self._inter_indices, self._inter_values,
            size=(self.config.n_neurons, self.config.n_neurons),
            device=self.device
        ).coalesce()

    def compute_field(self, xi: torch.Tensor) -> torch.Tensor:
        """Compute the total field h = W @ xi (intra + inter contributions).

        Args:
            xi: [n_neurons] activation pattern

        Returns:
            h: [n_neurons] field vector
        """
        bs = self.config.block_size

        # Reshape to blocks: [n_blocks, block_size]
        xi_blocks = xi.view(self.config.n_blocks, bs)

        # Intra-block field: batched dense matmul [n_blocks, block_size, block_size] @ [n_blocks, block_size, 1]
        h_intra = torch.bmm(self.W_intra, xi_blocks.unsqueeze(-1)).squeeze(-1)  # [n_blocks, block_size]
        h_intra = h_intra.view(-1)  # [n_neurons]

        # Inter-block field: sparse matmul
        W_inter = self._get_inter_sparse()
        h_inter = torch.sparse.mm(W_inter, xi.unsqueeze(-1)).squeeze(-1)  # [n_neurons]

        return h_intra + self.config.inter_block_weight * h_inter

    def energy(self, xi: torch.Tensor) -> float:
        """Compute Hopfield energy: E = -0.5 * xi^T @ W @ xi."""
        h = self.compute_field(xi)
        return -0.5 * torch.dot(xi, h).item()

    def update_step(self, xi: torch.Tensor) -> torch.Tensor:
        """Single attractor dynamics step.

        Args:
            xi: [n_neurons] current activation

        Returns:
            xi_new: [n_neurons] updated activation
        """
        h = self.compute_field(xi)

        if self.config.activation == "tanh":
            xi_new = torch.tanh(self.config.beta * h)
        elif self.config.activation == "softmax":
            # Block-wise softmax (maintain sparsity structure)
            bs = self.config.block_size
            h_blocks = h.view(self.config.n_blocks, bs)
            xi_blocks = F.softmax(self.config.beta * h_blocks, dim=1)
            xi_new = xi_blocks.view(-1)
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

        # Re-apply sparsity: keep only top-k activations by magnitude
        k = self.config.k_winners
        if k > 0 and k < self.config.n_neurons:
            topk_vals, topk_idx = torch.topk(xi_new.abs(), k)
            mask = torch.zeros_like(xi_new)
            mask[topk_idx] = 1.0
            xi_new = xi_new * mask

        return xi_new

    def attract(self, xi_0: torch.Tensor, anchor_strength: float = 0.3) -> tuple[torch.Tensor, list[float], int]:
        """Run attractor dynamics until convergence.

        Uses cue anchoring: each step blends the network's field with the original
        cue pattern. This prevents collapse to a single dominant attractor and keeps
        retrieval local to the cue's neighborhood in pattern space.

        Args:
            xi_0: [n_neurons] initial activation (the cue)
            anchor_strength: how much of the original cue to retain each step (0-1)

        Returns:
            xi_converged: [n_neurons] settled activation pattern
            energy_history: list of energy values per step
            n_steps: number of steps to convergence
        """
        xi = xi_0.clone()
        energy_history = []
        prev_energy = self.energy(xi)
        energy_history.append(prev_energy)

        for step in range(self.config.max_attractor_steps):
            xi_network = self.update_step(xi)
            # Anchor: blend network dynamics with original cue
            xi = (1 - anchor_strength) * xi_network + anchor_strength * xi_0

            # Re-apply sparsity after blending
            k = self.config.k_winners
            if k > 0 and k < self.config.n_neurons:
                topk_vals, topk_idx = torch.topk(xi.abs(), k)
                mask = torch.zeros_like(xi)
                mask[topk_idx] = 1.0
                xi = xi * mask

            current_energy = self.energy(xi)
            energy_history.append(current_energy)

            if abs(current_energy - prev_energy) < self.config.energy_epsilon:
                return xi, energy_history, step + 1

            prev_energy = current_energy

        return xi, energy_history, self.config.max_attractor_steps

    def state_dict(self) -> dict:
        return {
            "W_intra": self.W_intra.cpu(),
            "inter_indices": self._inter_indices.cpu(),
            "inter_values": self._inter_values.cpu(),
            "block_adjacency": self.block_adjacency,
        }

    def load_state_dict(self, state: dict):
        self.W_intra = state["W_intra"].to(self.device)
        self._inter_indices = state["inter_indices"].to(self.device)
        self._inter_values = state["inter_values"].to(self.device)
        self.block_adjacency = state["block_adjacency"]
