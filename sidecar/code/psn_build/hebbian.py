# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_hebbian.py
"""Hebbian learning: neurons that fire together wire together."""

import torch
from .config import PSNConfig
from .hopfield import HopfieldNetwork


class HebbianLearner:
    """Online Hebbian learning rule for the PSN.

    No gradients, no backprop, no optimizer. Pure co-activation strengthening:
        delta_W = eta * (xi_i * xi_j)

    For each stored thought:
        - Intra-block: outer product of active neurons within each block
        - Inter-block: outer product of active neurons across connected blocks
        - Weight clipping to prevent runaway growth
        - Periodic decay to mimic biological forgetting
    """

    def __init__(self, config: PSNConfig):
        self.config = config
        self.learn_count = 0

    @torch.no_grad()
    def learn(self, network: HopfieldNetwork, xi: torch.Tensor, active_indices: torch.Tensor):
        """Apply Hebbian update for a single stored pattern.

        Args:
            network: the Hopfield network to update
            xi: [n_neurons] sparse activation pattern
            active_indices: [k_winners] indices of active neurons
        """
        eta = self.config.eta
        bs = self.config.block_size

        # Reshape to blocks
        xi_blocks = xi.view(self.config.n_blocks, bs)

        # Find which blocks have active neurons
        active_blocks = set()
        for idx in active_indices.tolist():
            active_blocks.add(idx // bs)
        active_blocks = sorted(active_blocks)

        # Intra-block Hebbian update
        for b in active_blocks:
            xb = xi_blocks[b]  # [block_size]
            if xb.abs().sum() < 1e-8:
                continue
            # Outer product: delta_W = eta * xb @ xb^T
            delta = eta * torch.outer(xb, xb)
            # Symmetrize and zero diagonal
            delta = 0.5 * (delta + delta.T)
            delta.fill_diagonal_(0)
            network.W_intra[b] += delta

        # Inter-block Hebbian update
        self._update_inter(network, xi, active_blocks)

        # Normalize weights per block (prevent dominant basin collapse)
        self._normalize_weights(network, active_blocks)

        # Clip weights
        self._clip_weights(network)

        # Periodic decay
        self.learn_count += 1
        if self.learn_count % self.config.decay_interval == 0:
            self.decay(network)

    @torch.no_grad()
    def _update_inter(self, network: HopfieldNetwork, xi: torch.Tensor, active_blocks: list[int]):
        """Update inter-block connection weights via Hebbian rule."""
        if network._inter_values.numel() == 0:
            return

        eta = self.config.eta
        indices = network._inter_indices  # [2, n_connections]
        values = network._inter_values    # [n_connections]

        # For each existing inter-block connection, update weight
        # based on co-activation of the connected neurons
        src_activations = xi[indices[0]]  # [n_connections]
        dst_activations = xi[indices[1]]  # [n_connections]

        # Hebbian: strengthen connections between co-active neurons
        delta = eta * src_activations * dst_activations
        network._inter_values = values + delta

    @torch.no_grad()
    def _normalize_weights(self, network: HopfieldNetwork, active_blocks: list[int]):
        """Normalize weight matrices per block to prevent attractor dominance.

        Without normalization, frequently co-activated patterns create energy wells
        that absorb all queries. Frobenius normalization keeps each block's total
        synaptic strength bounded, ensuring multiple attractors can coexist.
        """
        target_norm = self.config.weight_clip  # target Frobenius norm per block
        for b in active_blocks:
            norm = network.W_intra[b].norm()
            if norm > target_norm:
                network.W_intra[b] *= target_norm / norm

    @torch.no_grad()
    def _clip_weights(self, network: HopfieldNetwork):
        """Clip weights to prevent unbounded growth."""
        max_w = self.config.weight_clip
        network.W_intra.clamp_(-max_w, max_w)
        network._inter_values.clamp_(-max_w, max_w)

    @torch.no_grad()
    def decay(self, network: HopfieldNetwork):
        """Apply weight decay: W *= (1 - decay_rate).

        Mimics biological synaptic weakening of unused connections.
        """
        factor = 1.0 - self.config.decay_rate
        network.W_intra *= factor
        network._inter_values *= factor

    @torch.no_grad()
    def prune_inter(self, network: HopfieldNetwork, threshold: float = 1e-5):
        """Remove near-zero inter-block connections to maintain sparsity."""
        mask = network._inter_values.abs() > threshold
        network._inter_indices = network._inter_indices[:, mask]
        network._inter_values = network._inter_values[mask]
