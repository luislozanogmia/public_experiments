# Tesseract

A 120K-parameter transformer that solves 10x10 mazes at 99% success rate across 500 unseen mazes. The key is the loss function: instead of averaging cross-entropy across all training examples, Tesseract decomposes training into 6 cells (navigation phase x decision type), enforces walkability and adjacency invariants per cell, and combines via smooth-max so the worst cell dominates gradient pressure. Same architecture trained with plain cross-entropy scores 22%.

## Results

| Model | Val Acc | Success (30 test) | Success (500 stress) | Adversarial (6 mazes) |
|---|---|---|---|---|
| Tesseract v5 | 69.5% | 100% | 99.0% | 67% (4/6) |
| CE Baseline | 64.9% | ~80% | 21.8% | 33% (2/6) |

## The Loss

Decomposes into 6 cells: (early/mid/endgame) x (corridor/fork). Per cell: smooth-max of walkability invariant, adjacency invariant, and cross-entropy. Cells combined via smooth-max (tau=0.10) - worst cell dominates. 0.05-weighted plain CE anchor for stability.

## Training

- Architecture: 3-layer transformer, 64 embed, 4 heads, 128 FFN dim, 120,356 params
- Data: 300 train mazes per seed, all-position BFS (~20 transitions/maze), 4 goal directions
- v5 loads v4 checkpoint and trains on a second seed for data diversity
- 240 epochs, cosine LR schedule, CPU trainable

## Run

```
python maze_tesseract.py          # train from scratch
python test_stress.py             # 500-maze stress test
python test_adversarial.py        # 6 adversarial mazes
cd viz && python maze_server.py   # browser UI at localhost:8765
```

## Requirements

torch, no GPU needed
