# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_config.py
"""PSN configuration - all hyperparameters in one place."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PSNConfig:
    # Network topology (200 neurons is sufficient - tested 50K→200 with same recall quality)
    n_neurons: int = 200
    n_blocks: int = 4
    block_size: int = 50  # n_neurons / n_blocks

    # Embedding
    d_embedding: int = 384  # all-MiniLM-L6-v2 output dim
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_cache_dir: str | None = None

    # Sparse encoding
    k_winners_pct: float = 0.02  # 2% of neurons active per thought
    k_winners: int = 0  # computed in __post_init__

    # Hopfield dynamics
    beta: float = 1.0  # inverse temperature for activation
    max_attractor_steps: int = 100
    energy_epsilon: float = 1e-6  # convergence threshold
    activation: str = "tanh"  # "tanh" or "softmax"

    # Inter-block connectivity
    inter_block_k: int = 10  # each block connects to K neighbor blocks
    inter_block_density: float = 0.05  # connection density within connected block pairs
    inter_block_weight: float = 0.1  # scaling factor for inter-block field

    # Hebbian learning
    eta: float = 0.01  # learning rate
    weight_clip: float = 1.0  # max absolute weight value
    decay_rate: float = 0.001  # per-decay-step weight decay
    decay_interval: int = 100  # apply decay every N learning events

    # Persistence
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    # Hardware (CPU is sufficient for 200 neurons - no GPU needed)
    device: str = "cpu"

    def __post_init__(self):
        assert self.n_neurons == self.n_blocks * self.block_size, \
            f"n_neurons ({self.n_neurons}) must equal n_blocks * block_size ({self.n_blocks * self.block_size})"
        self.k_winners = int(self.n_neurons * self.k_winners_pct)
