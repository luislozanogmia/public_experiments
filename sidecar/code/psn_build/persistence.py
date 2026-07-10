# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_persistence.py
"""Checkpoint save/load for the full PSN state.

Split architecture (v2):
  - psn_brain.pt  - weights, embeddings, neuron indices. No text. Safe to deploy.
  - psn_memory.jsonl - text indexed by ID. Stays local. Never uploaded.

"Your thoughts are local, your patterns are in the web."
"""

import json
from pathlib import Path
import torch
from .config import PSNConfig


PSN_CHECKPOINT_VERSION = 1


def _config_dict(config: PSNConfig) -> dict:
    return {
        "n_neurons": config.n_neurons,
        "n_blocks": config.n_blocks,
        "block_size": config.block_size,
        "d_embedding": config.d_embedding,
        "k_winners_pct": config.k_winners_pct,
        "beta": config.beta,
        "eta": config.eta,
        "decay_rate": config.decay_rate,
        "inter_block_k": config.inter_block_k,
        "inter_block_density": config.inter_block_density,
    }


def save_checkpoint(path: Path, config: PSNConfig, hopfield_state: dict,
                    projection_state: dict, memory_state: dict,
                    learner_count: int):
    """Save complete PSN state to disk (full checkpoint with text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": PSN_CHECKPOINT_VERSION,
        "config": _config_dict(config),
        "hopfield": hopfield_state,
        "projection": projection_state,
        "memory": memory_state,
        "learner_count": learner_count,
    }
    torch.save(checkpoint, path)


def _extract_concepts(text: str, max_keywords: int = 5) -> str:
    """Extract a short concept signature from text. No PII, no full sentences.

    Returns something like: 'AI architecture, validation, model training, reasoning'
    """
    import re
    if not text or len(text) < 10:
        return ""

    # Remove emails, URLs, numbers, signatures
    clean = re.sub(r'[\w.+-]+@[\w.-]+\.\w+', '', text)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'[+]?\d[\d\s\-()]{6,15}', '', clean)

    # Lowercase, split into words
    words = re.findall(r'[a-zA-Z\u00e0-\u00ff]{4,}', clean.lower())

    # Remove common stopwords
    stops = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'would',
             'could', 'should', 'about', 'what', 'when', 'where', 'which',
             'your', 'they', 'their', 'them', 'than', 'then', 'also', 'just',
             'like', 'very', 'some', 'more', 'most', 'much', 'many', 'each',
             'here', 'there', 'these', 'those', 'other', 'into', 'over',
             'after', 'before', 'between', 'under', 'again', 'does', 'doing',
             'being', 'having', 'para', 'como', 'pero', 'esto', 'esta',
             'esos', 'esas', 'hola', 'gracias', 'bien', 'bueno', 'puede',
             'porque', 'tambien', 'cuando', 'donde', 'solo', 'todo', 'todos',
             'tiene', 'hacer', 'creo', 'think', 'need', 'want', 'know'}
    words = [w for w in words if w not in stops]

    # Count frequency, take top N unique
    from collections import Counter
    counts = Counter(words)
    top = [w for w, _ in counts.most_common(max_keywords)]

    return ', '.join(top) if top else ""


def save_brain(path: Path, config: PSNConfig, hopfield_state: dict,
               projection_state: dict, memory_state: dict,
               learner_count: int):
    """Save brain-only checkpoint - NO TEXT. Safe to deploy publicly.

    Strips all text from memory entries. Stores short concept signatures
    instead (e.g., 'AI, architecture, validation'). Keeps embeddings,
    active_indices, timestamps, tags. Hopfield weights and projection intact.
    Attractor dynamics work. Embedding similarity works. Text recall returns
    concept signatures instead of full text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Deep copy memory state: replace text with concept signature
    safe_memory = {
        "next_id": memory_state["next_id"],
        "entries": {},
    }
    for eid, entry in memory_state["entries"].items():
        concepts = _extract_concepts(entry.get("text", ""))
        safe_memory["entries"][eid] = {
            "id": entry["id"],
            "text": concepts,  # concept signature, not full text
            "timestamp": entry.get("timestamp", 0),
            "embedding": entry["embedding"],
            "active_indices": entry["active_indices"],
            "retrieval_count": entry.get("retrieval_count", 0),
            "tags": entry.get("tags", []),
        }

    checkpoint = {
        "version": PSN_CHECKPOINT_VERSION,
        "config": _config_dict(config),
        "hopfield": hopfield_state,
        "projection": projection_state,
        "memory": safe_memory,
        "learner_count": learner_count,
        "brain_only": True,
    }
    torch.save(checkpoint, path)


def save_memory(path: Path, memory_state: dict):
    """Save text memory as JSONL - stays LOCAL, never uploaded.

    Each line: {"id": int, "text": str, "timestamp": ..., "tags": [...]}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for eid, entry in memory_state["entries"].items():
            record = {
                "id": entry["id"],
                "text": entry["text"],
                "timestamp": entry.get("timestamp", 0),
                "tags": entry.get("tags", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path) -> dict:
    """Load PSN checkpoint from disk (full or brain-only).

    Returns:
        dict with keys: version, config, hopfield, projection, memory, learner_count
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if checkpoint.get("version", 0) != PSN_CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version mismatch: expected {PSN_CHECKPOINT_VERSION}, "
                         f"got {checkpoint.get('version', 'unknown')}")

    return checkpoint


def load_memory(brain_checkpoint: dict, memory_path: Path) -> dict:
    """Rehydrate a brain-only checkpoint with text from a local memory file.

    Reads the JSONL memory file and patches text back into the checkpoint's
    memory state. Returns the patched checkpoint.
    """
    if not memory_path.exists():
        return brain_checkpoint  # no memory file, brain works without text

    # Build ID -> text lookup
    text_lookup = {}
    with open(memory_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text_lookup[record["id"]] = record["text"]

    # Patch text back into memory entries
    memory = brain_checkpoint.get("memory", {})
    patched = 0
    for eid, entry in memory.get("entries", {}).items():
        entry_id = entry.get("id", int(eid) if isinstance(eid, str) else eid)
        if entry_id in text_lookup:
            entry["text"] = text_lookup[entry_id]
            patched += 1

    return brain_checkpoint
