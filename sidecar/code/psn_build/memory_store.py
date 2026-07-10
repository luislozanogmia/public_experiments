# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_memory_store.py
"""Memory store: metadata index alongside synaptic patterns.

Optimized for scale: stores only embeddings (384d) and active neuron indices,
NOT full 50K dense activation vectors. At 86K thoughts this saves ~17GB RAM.
"""

import time
from dataclasses import dataclass, field
import torch


@dataclass
class MemoryEntry:
    """A single stored thought with metadata."""
    id: int
    text: str
    timestamp: float
    embedding: torch.Tensor  # [d_embedding] dense (384d = 1.5KB)
    active_indices: list[int]  # which neurons fired (~1000 ints = 4KB)
    retrieval_count: int = 0
    tags: list[str] = field(default_factory=list)


class MemoryStore:
    """Stores and retrieves thought metadata alongside the Hopfield network.

    The Hopfield network stores ASSOCIATIONS (weight patterns).
    The MemoryStore stores IDENTITIES (which thought produced which activation).

    Retrieval uses two strategies:
    - Embedding cosine similarity (384d, fast, primary)
    - Jaccard similarity on active neuron sets (set intersection, for attractor output)
    """

    def __init__(self):
        self.entries: dict[int, MemoryEntry] = {}
        self._next_id = 0
        self._embedding_matrix = None  # [N, d_embedding] cached
        self._cache_dirty = True

    def add(self, text: str, embedding: torch.Tensor, activation: torch.Tensor,
            active_indices: torch.Tensor, tags: list[str] = None) -> int:
        """Store a new thought (sparse - only embedding + active indices)."""
        entry = MemoryEntry(
            id=self._next_id,
            text=text,
            timestamp=time.time(),
            embedding=embedding.cpu().half(),  # fp16 saves 50% RAM
            active_indices=active_indices.cpu().tolist() if isinstance(active_indices, torch.Tensor) else list(active_indices),
            tags=tags or [],
        )
        self.entries[self._next_id] = entry
        self._next_id += 1
        self._cache_dirty = True
        return entry.id

    def get(self, memory_id: int) -> MemoryEntry | None:
        return self.entries.get(memory_id)

    def find_nearest(self, activation: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        """Find nearest stored patterns using Jaccard similarity on active neuron sets.

        Given a converged attractor activation, find which stored patterns share
        the most active neurons (set intersection / set union).
        Memory-efficient: no 50K-dim dense vectors needed.
        """
        if not self.entries:
            return []

        # Get active indices from the query activation
        k_query = min(1000, (activation > 0).sum().item())
        if k_query == 0:
            return self.find_nearest_by_embedding(activation[:384] if activation.shape[0] > 384 else activation, top_k)

        _, query_indices = torch.topk(activation.abs(), k_query)
        query_set = set(query_indices.cpu().tolist())

        # Jaccard similarity against all stored patterns
        scores = []
        for eid, entry in self.entries.items():
            stored_set = set(entry.active_indices)
            intersection = len(query_set & stored_set)
            union = len(query_set | stored_set)
            jaccard = intersection / union if union > 0 else 0.0
            scores.append((eid, jaccard))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def find_nearest_by_embedding(self, embedding: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        """Find nearest by dense embedding similarity (384d, fast)."""
        if not self.entries:
            return []

        if self._cache_dirty:
            self._rebuild_cache()

        query = embedding.cpu().float().unsqueeze(0)  # [1, 384]
        sims = torch.nn.functional.cosine_similarity(
            query, self._embedding_matrix.float(), dim=1
        )

        k = min(top_k, len(self.entries))
        topk_sims, topk_idx = torch.topk(sims, k)

        ids = list(self.entries.keys())
        return [(ids[topk_idx[i].item()], topk_sims[i].item()) for i in range(k)]

    def _rebuild_cache(self):
        """Rebuild embeddings matrix cache."""
        if self.entries:
            self._embedding_matrix = torch.stack([e.embedding for e in self.entries.values()])
        else:
            self._embedding_matrix = torch.empty(0)
        self._cache_dirty = False

    @property
    def count(self) -> int:
        return len(self.entries)

    def metadata_dict(self) -> list[dict]:
        """Export metadata (without tensors) for inspection."""
        return [
            {
                "id": e.id,
                "text": e.text,
                "timestamp": e.timestamp,
                "n_active_neurons": len(e.active_indices),
                "retrieval_count": e.retrieval_count,
                "tags": e.tags,
            }
            for e in self.entries.values()
        ]

    def state_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "entries": {
                eid: {
                    "id": e.id,
                    "text": e.text,
                    "timestamp": e.timestamp,
                    "embedding": e.embedding,
                    "active_indices": e.active_indices,
                    "retrieval_count": e.retrieval_count,
                    "tags": e.tags,
                }
                for eid, e in self.entries.items()
            }
        }

    def load_state_dict(self, state: dict):
        self._next_id = state["next_id"]
        self.entries = {}
        for eid_str, data in state["entries"].items():
            eid = int(eid_str) if isinstance(eid_str, str) else eid_str
            emb = data["embedding"]
            if isinstance(emb, torch.Tensor) and emb.dtype != torch.float16:
                emb = emb.half()
            self.entries[eid] = MemoryEntry(
                id=data["id"],
                text=data["text"],
                timestamp=data["timestamp"],
                embedding=emb,
                active_indices=data["active_indices"],
                retrieval_count=data.get("retrieval_count", 0),
                tags=data.get("tags", []),
            )
        self._cache_dirty = True
