"""Build a PSN checkpoint from a JSONL corpus.

Reads a corpus file (one JSON object per line, each with a "text" field and
an optional "tags" field), stores every entry into a fresh PSN, runs a couple
of recall sanity checks, and saves the resulting checkpoint to disk.

Usage (from the sidecar/ directory):
    python code/build_psn.py --corpus data/sample_corpus.jsonl --neurons 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from psn_build.config import PSNConfig
from psn_build.psn import PSN

DEFAULT_CORPUS = Path(__file__).parent.parent / "data" / "sample_corpus.jsonl"
DEFAULT_OUT = Path(__file__).parent / "checkpoints" / "psn_latest.pt"

# A couple of recall sanity-check queries, similar in spirit to the sample corpus
SANITY_QUERIES = [
    "I prefer small careful experiments over big risky ones",
    "trust measurements more than gut feelings",
]


def load_corpus(path: Path) -> list[dict]:
    """Load a JSONL corpus file, skipping blank lines."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_config(neurons: int) -> PSNConfig:
    """Build a PSNConfig with n_neurons/n_blocks/block_size kept consistent.

    Keeps n_blocks at the dataclass default (4) and derives block_size from
    the requested neuron count. Raises if the division isn't exact, since
    PSNConfig.__post_init__ asserts n_neurons == n_blocks * block_size.
    """
    default_blocks = PSNConfig.n_blocks
    if neurons % default_blocks != 0:
        raise ValueError(
            f"--neurons ({neurons}) must be evenly divisible by n_blocks "
            f"({default_blocks}), got remainder {neurons % default_blocks}"
        )
    block_size = neurons // default_blocks
    return PSNConfig(n_neurons=neurons, n_blocks=default_blocks, block_size=block_size)


def main():
    parser = argparse.ArgumentParser(description="Ingest a JSONL corpus into a fresh PSN and save a checkpoint.")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="path to a JSONL corpus file")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="path to save the PSN checkpoint")
    parser.add_argument("--neurons", type=int, default=PSNConfig.n_neurons, help="number of PSN neurons")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_path = Path(args.out)

    config = build_config(args.neurons)

    print(f"Corpus: {corpus_path}")
    print(f"Output: {out_path}")
    print(f"Neurons: {config.n_neurons} (n_blocks={config.n_blocks}, block_size={config.block_size})")

    t0 = time.time()
    psn = PSN(config)
    print(f"PSN instantiated in {time.time() - t0:.2f}s (includes downloading the encoder model if not cached)")

    records = load_corpus(corpus_path)
    print(f"\nLoaded {len(records)} corpus entries from {corpus_path.name}")

    for i, rec in enumerate(records):
        result = psn.store(rec["text"], tags=rec.get("tags"))
        print(f"  [{i}] id={result['id']} active_neurons={result['n_active_neurons']} "
              f"energy={result['energy']:.4f} elapsed_ms={result['elapsed_ms']:.2f}")

    print("\n=== PSN STATUS AFTER INGESTING CORPUS ===")
    status = psn.status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n=== RECALL SANITY CHECKS ===")
    for query in SANITY_QUERIES:
        print(f"\nQuery: {query}")
        recall = psn.recall(query, top_k=3)
        for match in recall["matches"]:
            print(f"  id={match['id']} sim={match['similarity']} text={match['text']}")

    # psn.save() -> persistence.save_checkpoint() already creates parent dirs
    saved_path = psn.save(out_path)
    print(f"\nSaved checkpoint to: {saved_path}")


if __name__ == "__main__":
    main()
