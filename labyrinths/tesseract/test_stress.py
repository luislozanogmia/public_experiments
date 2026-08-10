"""
Stress test: Tesseract v5 checkpoint across 5 random seeds.
Verifies the 100% success rate holds beyond the original test seed.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Intro-NN", "labs"))

from maze_tesseract_v5_cpu import (
    LabyrinthTransformer,
    generate_maze_set,
    evaluate_model,
    DEVICE,
)

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "maze_tesseract_v5.pt")
SEEDS = [999, 2024, 7777, 31415, 54321]
MAZES_PER_SEED = 100


def main():
    model = LabyrinthTransformer(embed_dim=64, num_heads=4, hidden_dim=128, num_layers=3).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}", flush=True)

    per_seed_results = []
    total_successes = 0
    total_mazes = 0

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        mazes = generate_maze_set(
            MAZES_PER_SEED, seed=seed,
            corner_offset=(0, 3),
            num_loops_range=(4, 8),
        )
        print(f"  Generated {len(mazes)} mazes", flush=True)
        metrics = evaluate_model(model, mazes, DEVICE)
        print(f"  Success Rate:   {metrics['success_rate']:.2f}% ({metrics['successes']}/{metrics['total']})", flush=True)
        print(f"  Optimal Rate:   {metrics['optimal_rate']:.2f}%", flush=True)
        print(f"  Avg Efficiency: {metrics['avg_efficiency']:.2f}%", flush=True)

        per_seed_results.append((seed, metrics))
        total_successes += metrics["successes"]
        total_mazes += metrics["total"]

    print(f"\n{'=' * 60}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    for seed, metrics in per_seed_results:
        print(f"  Seed {seed:>6}: {metrics['success_rate']:.2f}% ({metrics['successes']}/{metrics['total']})", flush=True)
    overall = total_successes / total_mazes * 100
    print(f"\n  Overall: {overall:.2f}% ({total_successes}/{total_mazes}) across {len(SEEDS)} seeds", flush=True)


if __name__ == "__main__":
    main()
