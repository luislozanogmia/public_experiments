# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_psn.py
"""PSN: the orchestrator. Store thoughts, retrieve associations."""

import time
from pathlib import Path
import torch
from .config import PSNConfig
from .encoder import TextEncoder
from .projection import SparseProjection
from .hopfield import HopfieldNetwork
from .hebbian import HebbianLearner
from .memory_store import MemoryStore, MemoryEntry
from .persistence import save_checkpoint, load_checkpoint, save_brain, save_memory, load_memory


class PSN:
    """Personal Synaptic Network - store and retrieve one person's thought patterns.

    Public API:
        store(text)             - encode a thought and learn it into the network
        recall(cue, top_k)      - retrieve associated patterns from a cue
        status()                - network stats
        save(path) / load(path) - persistence
    """

    def __init__(self, config: PSNConfig = None):
        self.config = config or PSNConfig()
        self.encoder = TextEncoder(self.config)
        self.projection = SparseProjection(self.config)
        self.network = HopfieldNetwork(self.config)
        self.learner = HebbianLearner(self.config)
        self.memory = MemoryStore()

    def store(self, text: str, tags: list[str] = None) -> dict:
        """Store a thought into the synaptic network.

        1. Encode text -> 384d embedding
        2. Project to sparse 50K activation via k-WTA
        3. Hebbian update: strengthen co-active connections
        4. Save metadata

        Returns:
            dict with: id, n_active_neurons, energy, learn_count
        """
        t0 = time.perf_counter()

        # Encode
        embedding = self.encoder.encode(text)

        # Project to sparse activation
        activation, active_indices = self.projection.project(embedding)

        # Hebbian learning
        self.learner.learn(self.network, activation, active_indices)

        # Store metadata
        memory_id = self.memory.add(text, embedding, activation, active_indices, tags)

        # Compute energy at the new pattern
        energy = self.network.energy(activation)

        elapsed = time.perf_counter() - t0

        return {
            "id": memory_id,
            "n_active_neurons": len(active_indices),
            "energy": energy,
            "learn_count": self.learner.learn_count,
            "elapsed_ms": elapsed * 1000,
        }

    def recall(self, cue: str, top_k: int = 5, use_attractor: bool = True) -> dict:
        """Retrieve associated patterns from a text cue.

        1. Encode cue -> 384d embedding
        2. Project to sparse activation
        3. (Optional) Run attractor dynamics to pattern completion
        4. Find nearest stored patterns

        Returns:
            dict with: matches (list of {id, text, similarity, tags}),
                       n_steps, energy_history, elapsed_ms
        """
        t0 = time.perf_counter()

        # Encode cue
        embedding = self.encoder.encode(cue)

        # Project
        activation, _ = self.projection.project(embedding)

        # Attractor dynamics
        if use_attractor and self.memory.count > 0:
            converged, energy_history, n_steps = self.network.attract(activation)
        else:
            converged = activation
            energy_history = [self.network.energy(activation)]
            n_steps = 0

        # Primary: embedding-based matching (fast, scales to 86K+)
        embed_matches = self.memory.find_nearest_by_embedding(embedding, top_k)

        # Secondary: Jaccard similarity on attractor output (associative enhancement)
        matches_raw = self.memory.find_nearest(converged, top_k) if use_attractor else []

        # Format results - embedding matches are primary
        matches = []
        for memory_id, sim in embed_matches:
            entry = self.memory.get(memory_id)
            if entry:
                entry.retrieval_count += 1
                matches.append({
                    "id": entry.id,
                    "text": entry.text,
                    "similarity": round(sim, 4),
                    "tags": entry.tags,
                    "retrieval_count": entry.retrieval_count,
                })

        # Jaccard matches from attractor (secondary, diagnostic)
        jaccard_results = []
        for memory_id, sim in matches_raw:
            entry = self.memory.get(memory_id)
            if entry:
                jaccard_results.append({
                    "id": entry.id,
                    "text": entry.text,
                    "similarity": round(sim, 4),
                })

        elapsed = time.perf_counter() - t0

        return {
            "matches": matches,
            "jaccard_matches": jaccard_results,
            "n_steps": n_steps,
            "energy_start": energy_history[0] if energy_history else 0,
            "energy_final": energy_history[-1] if energy_history else 0,
            "elapsed_ms": elapsed * 1000,
        }

    def status(self) -> dict:
        """Network statistics."""
        # Weight norms per block
        intra_norms = self.network.W_intra.norm(dim=(1, 2)).cpu().tolist()
        avg_intra_norm = sum(intra_norms) / len(intra_norms) if intra_norms else 0

        # Inter-block stats
        n_inter = self.network._inter_values.numel()
        inter_nonzero = (self.network._inter_values.abs() > 1e-8).sum().item()

        # Memory usage estimate (bytes)
        intra_bytes = self.network.W_intra.numel() * 4  # float32
        inter_bytes = (self.network._inter_indices.numel() + self.network._inter_values.numel()) * 4
        proj_bytes = self.projection.W_proj.numel() * 4
        total_mb = (intra_bytes + inter_bytes + proj_bytes) / 1e6

        return {
            "n_neurons": self.config.n_neurons,
            "n_blocks": self.config.n_blocks,
            "block_size": self.config.block_size,
            "k_winners": self.config.k_winners,
            "n_stored_patterns": self.memory.count,
            "learn_count": self.learner.learn_count,
            "avg_intra_weight_norm": round(avg_intra_norm, 4),
            "inter_connections_total": n_inter,
            "inter_connections_active": inter_nonzero,
            "memory_mb": round(total_mb, 1),
            "device": self.config.device,
        }

    def save(self, path: str | Path = None):
        """Save full network state to checkpoint (includes text)."""
        if path is None:
            path = self.config.checkpoint_dir / "psn_latest.pt"
        path = Path(path)

        save_checkpoint(
            path=path,
            config=self.config,
            hopfield_state=self.network.state_dict(),
            projection_state=self.projection.state_dict(),
            memory_state=self.memory.state_dict(),
            learner_count=self.learner.learn_count,
        )
        return str(path)

    def save_split(self, brain_path: str | Path = None, memory_path: str | Path = None):
        """Save as split: brain (no text, safe to deploy) + memory (text, stays local).

        'Your thoughts are local, your patterns are in the web.'
        """
        if brain_path is None:
            brain_path = self.config.checkpoint_dir / "psn_brain.pt"
        if memory_path is None:
            memory_path = self.config.checkpoint_dir / "psn_memory.jsonl"
        brain_path = Path(brain_path)
        memory_path = Path(memory_path)

        mem_state = self.memory.state_dict()

        save_brain(
            path=brain_path,
            config=self.config,
            hopfield_state=self.network.state_dict(),
            projection_state=self.projection.state_dict(),
            memory_state=mem_state,
            learner_count=self.learner.learn_count,
        )

        save_memory(path=memory_path, memory_state=mem_state)

        return str(brain_path), str(memory_path)

    def load(self, path: str | Path, memory_path: str | Path = None):
        """Load network state from checkpoint.

        If memory_path is provided, rehydrates text from the local memory file.
        If the checkpoint is brain-only and no memory_path given, text will be empty
        but attractor dynamics and embedding matching still work.
        """
        path = Path(path)
        checkpoint = load_checkpoint(path)

        # Rehydrate text if memory file explicitly provided
        if memory_path:
            checkpoint = load_memory(checkpoint, Path(memory_path))

        self.network.load_state_dict(checkpoint["hopfield"])
        self.projection.load_state_dict(checkpoint["projection"])
        self.memory.load_state_dict(checkpoint["memory"])
        self.learner.learn_count = checkpoint.get("learner_count", 0)

    def decay(self):
        """Manually trigger weight decay (forgetting)."""
        self.learner.decay(self.network)

    def inspect_block(self, block_id: int) -> dict:
        """Inspect a specific block's state."""
        W = self.network.W_intra[block_id]
        return {
            "block_id": block_id,
            "weight_norm": W.norm().item(),
            "weight_mean": W.mean().item(),
            "weight_max": W.max().item(),
            "weight_min": W.min().item(),
            "n_nonzero": (W.abs() > 1e-8).sum().item(),
            "neighbors": self.network.block_adjacency.get(block_id, []),
        }
