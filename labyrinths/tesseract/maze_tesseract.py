"""
Maze Tesseract v5: CPU, new seed mazes, continued from v4 checkpoint
====================================================
v5 changes from v4, and nothing else (one variable per the one-hypothesis
rule): SEED changed to 137 so an entirely new set of train/val/test mazes
is generated, and the model is initialized from the v4 checkpoint instead
of from scratch (fresh optimizer, so the cosine annealing schedule
restarts over the same EPOCHS horizon). Same EPOCHS, same CPU device,
same LR_TESSERACT, same Tesseract loss, same architecture. Saved as
maze_tesseract_v5.pt, v4 checkpoint kept intact.
"""

import math
import os
import sys
import json
import random
import time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Import external generator ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Intro-NN", "labs"))
from solver import generate_labyrinth, solve_bfs

# ── Constants ───────────────────────────────────────────────────────
GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE
VOCAB_SIZE = 10
WALKABLE_TOKENS = {0, 1, 2}

NAV_PHASES = ("early", "mid", "endgame")
DECISION_TYPES = ("corridor", "fork")
INVARIANT_NAMES = ("walkability", "adjacency")

SEED = 137
NUM_TRAIN_MAZES = 300
NUM_VAL_MAZES = 50
NUM_TEST_MAZES = 30
BATCH_SIZE = 64
EPOCHS = 240  # 3x v3's 80, targeting the still-climbing curve to actually converge
LR = 1e-3
LR_TESSERACT = 2e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"  # forced: RTX reserved, retrain must not touch it

MAX_EVAL_STEPS = 50
REVISIT_PENALTY = 0.01


# ── Model (external architecture) ────────────────────────────────

class LabyrinthTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=3):
        super().__init__()
        self.grid_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.pos_embedding = nn.Embedding(NUM_CELLS, embed_dim)
        self.spatial_embedding = nn.Parameter(torch.randn(1, NUM_CELLS, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, NUM_CELLS)

    def forward(self, grid, curr_pos):
        grid_emb = self.grid_embedding(grid) + self.spatial_embedding
        pos_emb = self.pos_embedding(curr_pos).unsqueeze(1)
        x = grid_emb + pos_emb
        out = self.transformer(x)
        batch_idx = torch.arange(grid.size(0), device=grid.device)
        return self.fc_out(out[batch_idx, curr_pos])


# ── Tesseract primitives ───────────────────────────────────────────

def smoothmax(values, tau=0.10):
    if not values:
        raise ValueError("smooth-max requires at least one value")
    if len(values) == 1:
        return values[0]
    stack = torch.stack(values)
    return tau * (torch.logsumexp(stack / tau, dim=0) - math.log(len(values)))


def smoothmax_1d(values, tau=0.15):
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    if values.numel() == 1:
        return values[0]
    return tau * (torch.logsumexp(values / tau, dim=0) - math.log(values.numel()))


# ── BFS for all positions ──────────────────────────────────────────

def bfs_all_next_steps(grid_2d, end):
    """
    BFS backward from end. For every walkable cell, compute the optimal
    next step toward end. Returns dict[(r,c)] -> (next_r, next_c) and
    dict[(r,c)] -> distance_to_end.
    """
    er, ec = end
    dist = {}
    parent = {}  # parent[cell] = the cell that discovered it (i.e., one step closer to end)
    queue = deque()

    dist[(er, ec)] = 0
    queue.append((er, ec))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if grid_2d[nr][nc] in WALKABLE_TOKENS and (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    # For each cell, "next step toward end" = parent[cell]
    # (since BFS was from end, parent points one step closer to end)
    next_step = {}
    for cell in parent:
        next_step[cell] = parent[cell]
    # end cell has no next step (it IS the goal)

    return next_step, dist


# ── Metadata tagging ───────────────────────────────────────────────

def assign_phase_by_distance(distance, max_distance):
    """Phase by BFS distance to goal: endgame=close, early=far."""
    if max_distance == 0:
        return 2
    ratio = distance / max_distance
    if ratio > 0.66:
        return 0  # early (far from goal)
    elif ratio > 0.33:
        return 1  # mid
    else:
        return 2  # endgame (close to goal)


def classify_decision(grid_2d, r, c):
    walkable_neighbors = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            if grid_2d[nr][nc] in WALKABLE_TOKENS:
                walkable_neighbors += 1
    return 0 if walkable_neighbors <= 2 else 1


def build_walkable_mask(flat_grid):
    return torch.tensor([v in WALKABLE_TOKENS for v in flat_grid], dtype=torch.bool)


def build_adjacent_mask(curr_idx):
    mask = torch.zeros(NUM_CELLS, dtype=torch.bool)
    r, c = curr_idx // GRID_SIZE, curr_idx % GRID_SIZE
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            mask[nr * GRID_SIZE + nc] = True
    return mask


def flatten_grid(grid_2d):
    return [grid_2d[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


# ── Data pipeline ──────────────────────────────────────────────────

# Direction pairs: start corner -> end corner
CORNER_PAIRS = [
    ("top-left", "bottom-right"),
    ("bottom-right", "top-left"),
    ("top-right", "bottom-left"),
    ("bottom-left", "top-right"),
]


def pick_corner_pos(corner, offset_lo, offset_hi):
    """Pick (row, col) in a corner of the grid.
    offset_lo/hi = distance from the corner edge."""
    if corner.startswith("top"):
        r = random.randint(offset_lo, offset_hi)
    else:
        r = random.randint(GRID_SIZE - 1 - offset_hi, GRID_SIZE - 1 - offset_lo)
    if corner.endswith("left"):
        c = random.randint(offset_lo, offset_hi)
    else:
        c = random.randint(GRID_SIZE - 1 - offset_hi, GRID_SIZE - 1 - offset_lo)
    return (r, c)


def generate_maze_set(num_mazes, seed, corner_offset, num_loops_range):
    random.seed(seed)
    mazes = []
    dir_counts = defaultdict(int)
    attempts = 0
    max_attempts = num_mazes * 15
    while len(mazes) < num_mazes and attempts < max_attempts:
        attempts += 1
        start_corner, end_corner = random.choice(CORNER_PAIRS)
        sr = pick_corner_pos(start_corner, *corner_offset)
        er = pick_corner_pos(end_corner, *corner_offset)
        nl = random.randint(*num_loops_range)
        try:
            grid = generate_labyrinth(GRID_SIZE, GRID_SIZE, sr, er, num_loops=nl)
        except Exception:
            continue
        path = solve_bfs(grid, sr, er)
        if path and len(path) > 5:
            mazes.append((grid, sr, er, path))
            dir_counts[f"{start_corner}->{end_corner}"] += 1
    print(f"  Direction distribution: {dict(dir_counts)}", flush=True)
    return mazes


def build_all_position_transitions(mazes):
    """
    For every maze, for every walkable cell (except end), compute the
    BFS-optimal next step toward end. This gives the model training signal
    from every reachable position, not just the optimal path.
    """
    transitions = []
    for grid_2d, start, end, opt_path in mazes:
        flat = flatten_grid(grid_2d)
        next_steps, distances = bfs_all_next_steps(grid_2d, end)

        if not distances:
            continue
        max_dist = max(distances.values())

        for (r, c), (nr, nc) in next_steps.items():
            curr_idx = r * GRID_SIZE + c
            next_idx = nr * GRID_SIZE + nc
            dist = distances[(r, c)]

            transitions.append({
                "grid": flat,
                "curr_idx": curr_idx,
                "next_idx": next_idx,
                "phase": assign_phase_by_distance(dist, max_dist),
                "decision": classify_decision(grid_2d, r, c),
                "walkable_mask": build_walkable_mask(flat),
                "adjacent_mask": build_adjacent_mask(curr_idx),
            })
    return transitions


class MazeDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            "grid": torch.tensor(t["grid"], dtype=torch.long),
            "curr_idx": torch.tensor(t["curr_idx"], dtype=torch.long),
            "next_idx": torch.tensor(t["next_idx"], dtype=torch.long),
            "phase": torch.tensor(t["phase"], dtype=torch.long),
            "decision": torch.tensor(t["decision"], dtype=torch.long),
            "walkable_mask": t["walkable_mask"],
            "adjacent_mask": t["adjacent_mask"],
        }


# ── Tesseract loss ─────────────────────────────────────────────────

def compute_invariant_losses(logits, walkable_masks, adjacent_masks):
    probs = F.softmax(logits, dim=-1)
    walkable_mass = (probs * walkable_masks.float()).sum(dim=-1)
    walkability_loss = -torch.log(walkable_mass.clamp_min(1e-8))
    adj_walkable = walkable_masks & adjacent_masks
    adj_mass = (probs * adj_walkable.float()).sum(dim=-1)
    adjacency_loss = -torch.log(adj_mass.clamp_min(1e-8))
    return {"walkability": walkability_loss, "adjacency": adjacency_loss}


def maze_tesseract_loss(logits, targets, batch_meta):
    ce_per_example = F.cross_entropy(logits, targets, reduction="none")
    invariants = compute_invariant_losses(
        logits, batch_meta["walkable_mask"], batch_meta["adjacent_mask"]
    )
    phases = batch_meta["phase"]
    decisions = batch_meta["decision"]

    cell_losses = {}
    for p_idx, phase_name in enumerate(NAV_PHASES):
        for d_idx, decision_name in enumerate(DECISION_TYPES):
            active = (phases == p_idx) & (decisions == d_idx)
            if not active.any():
                continue
            content_reduced = smoothmax_1d(ce_per_example[active], tau=0.15)
            inv_list = [smoothmax_1d(invariants[k][active], tau=0.15) for k in INVARIANT_NAMES]
            baseline_gate = smoothmax(inv_list, tau=0.10)
            cell_losses[(phase_name, decision_name)] = smoothmax(
                [baseline_gate, content_reduced], tau=0.10
            )

    if not cell_losses:
        return ce_per_example.mean()

    protected = smoothmax(list(cell_losses.values()), tau=0.10)
    ce_anchor = ce_per_example.mean()
    return protected + 0.05 * ce_anchor


# ── Training ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, use_tesseract=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        grid = batch["grid"].to(device)
        curr = batch["curr_idx"].to(device)
        target = batch["next_idx"].to(device)
        logits = model(grid, curr)

        if use_tesseract:
            meta = {k: batch[k].to(device) for k in
                    ["phase", "decision", "walkable_mask", "adjacent_mask"]}
            loss = maze_tesseract_loss(logits, target, meta)
        else:
            loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * grid.size(0)
        total_correct += (logits.argmax(dim=-1) == target).sum().item()
        total_samples += grid.size(0)

    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            grid = batch["grid"].to(device)
            curr = batch["curr_idx"].to(device)
            target = batch["next_idx"].to(device)
            logits = model(grid, curr)
            loss = F.cross_entropy(logits, target)
            total_loss += loss.item() * grid.size(0)
            total_correct += (logits.argmax(dim=-1) == target).sum().item()
            total_samples += grid.size(0)
    return total_loss / total_samples, total_correct / total_samples


def compute_cell_metrics(model, loader, device):
    model.eval()
    cell_correct = defaultdict(int)
    cell_total = defaultdict(int)
    with torch.no_grad():
        for batch in loader:
            grid = batch["grid"].to(device)
            curr = batch["curr_idx"].to(device)
            target = batch["next_idx"].to(device)
            phases = batch["phase"]
            decisions = batch["decision"]
            preds = model(grid, curr).argmax(dim=-1)
            correct = (preds == target).cpu()
            for i in range(grid.size(0)):
                key = (NAV_PHASES[phases[i].item()], DECISION_TYPES[decisions[i].item()])
                cell_total[key] += 1
                cell_correct[key] += correct[i].item()
    return {k: cell_correct[k] / cell_total[k] if cell_total[k] > 0 else 0.0 for k in cell_total}


# ── Eval ───────────────────────────────────────────────────────────

def solve_autoregressive(model, grid_2d, start, end, device, max_steps=MAX_EVAL_STEPS):
    """Prefer unvisited neighbors (model-ranked). If all visited, backtrack to
    least-recently-visited. Eliminates oscillation without retraining."""
    model.eval()
    flat = flatten_grid(grid_2d)
    grid_t = torch.tensor(flat, dtype=torch.long, device=device).unsqueeze(0)
    curr_r, curr_c = start
    path = [(curr_r, curr_c)]
    visited = {(curr_r, curr_c)}
    visit_order = {(curr_r, curr_c): 0}
    step = 0
    with torch.no_grad():
        for _ in range(max_steps):
            if (curr_r, curr_c) == end:
                break
            step += 1
            curr_idx = curr_r * GRID_SIZE + curr_c
            curr_t = torch.tensor(curr_idx, dtype=torch.long, device=device).unsqueeze(0)
            probs = F.softmax(model(grid_t, curr_t), dim=-1).squeeze(0)
            unvisited = []
            revisitable = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    if grid_2d[nr][nc] in WALKABLE_TOKENS:
                        p = probs[nr * GRID_SIZE + nc].item()
                        if (nr, nc) not in visited:
                            unvisited.append((p, nr, nc))
                        else:
                            revisitable.append((visit_order[(nr, nc)], p, nr, nc))
            if unvisited:
                unvisited.sort(key=lambda x: -x[0])
                _, curr_r, curr_c = unvisited[0]
            elif revisitable:
                revisitable.sort(key=lambda x: x[0])
                _, _, curr_r, curr_c = revisitable[0]
            else:
                break
            path.append((curr_r, curr_c))
            visited.add((curr_r, curr_c))
            visit_order[(curr_r, curr_c)] = step
    return path


def evaluate_model(model, test_mazes, device):
    successes = optimal = 0
    efficiencies = []
    for grid_2d, start, end, opt_path in test_mazes:
        actual = solve_autoregressive(model, grid_2d, start, end, device)
        if actual[-1] == end:
            successes += 1
            eff = len(opt_path) / len(actual) * 100
            efficiencies.append(eff)
            if len(actual) == len(opt_path):
                optimal += 1
    n = len(test_mazes)
    return {
        "success_rate": successes / n * 100,
        "optimal_rate": optimal / n * 100,
        "avg_efficiency": np.mean(efficiencies) if efficiencies else 0.0,
        "successes": successes, "total": n,
    }


# ── Main ───────────────────────────────────────────────────────────

def run_experiment():
    print(f"{'=' * 70}", flush=True)
    print(f"  Tesseract v5 (seed=137, continuing from v4)", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── Generate mazes ──
    print("[1/7] Generating mazes...", flush=True)
    train_mazes = generate_maze_set(
        NUM_TRAIN_MAZES, seed=SEED,
        corner_offset=(0, 3),
        num_loops_range=(4, 8),
    )
    val_mazes = generate_maze_set(
        NUM_VAL_MAZES, seed=SEED + 500,
        corner_offset=(0, 3),
        num_loops_range=(4, 8),
    )
    test_mazes = generate_maze_set(
        NUM_TEST_MAZES, seed=SEED + 1000,
        corner_offset=(0, 2),
        num_loops_range=(6, 6),
    )
    print(f"  Train: {len(train_mazes)} mazes", flush=True)
    print(f"  Val:   {len(val_mazes)} mazes (separate, for early stopping)", flush=True)
    print(f"  Test:  {len(test_mazes)} mazes (held out)", flush=True)

    # ── Build transitions from ALL walkable positions ──
    print("[2/7] Building all-position transitions...", flush=True)
    train_transitions = build_all_position_transitions(train_mazes)
    val_transitions = build_all_position_transitions(val_mazes)
    random.shuffle(train_transitions)

    train_dataset = MazeDataset(train_transitions)
    val_dataset = MazeDataset(val_transitions)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Stats
    cell_counts = defaultdict(int)
    for t in train_transitions:
        key = (NAV_PHASES[t["phase"]], DECISION_TYPES[t["decision"]])
        cell_counts[key] += 1

    print(f"  Train transitions: {len(train_transitions)} (from {len(train_mazes)} mazes)", flush=True)
    print(f"  Val transitions:   {len(val_transitions)} (from {len(val_mazes)} mazes)", flush=True)
    avg_per_maze = len(train_transitions) / len(train_mazes)
    print(f"  Avg per maze: {avg_per_maze:.1f} (vs ~14 in v1 optimal-path-only)", flush=True)
    print(f"  Cell distribution:", flush=True)
    for key in sorted(cell_counts.keys()):
        print(f"    {key}: {cell_counts[key]}", flush=True)
    print(flush=True)

    # ── Train Tesseract only. CE v3 already converged, left untouched. ──
    results = {}

    for mode in ["tesseract"]:
        use_tesseract = mode == "tesseract"
        step_n = "5"
        print(f"[{step_n}/7] Training {mode} (v5: {EPOCHS} epochs, CPU)...", flush=True)

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        model = LabyrinthTransformer().to(DEVICE)
        v4_path = os.path.join(os.path.dirname(__file__), "checkpoints", "maze_tesseract_v4.pt")
        model.load_state_dict(torch.load(v4_path, map_location=DEVICE))
        print(f"  Loaded v4 checkpoint: {v4_path}", flush=True)
        lr = LR_TESSERACT if use_tesseract else LR
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}  |  LR: {lr}", flush=True)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            t_loss, t_acc = train_epoch(model, train_loader, optimizer, DEVICE, use_tesseract)
            v_loss, v_acc = eval_epoch(model, val_loader, DEVICE)
            scheduler.step()

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:02d}/{EPOCHS} | "
                    f"Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.1f}% | "
                    f"Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.1f}%",
                    flush=True,
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        cell_metrics = compute_cell_metrics(model, val_loader, DEVICE)
        print(f"  Best val acc: {best_val_acc*100:.1f}%", flush=True)
        print(f"  Per-cell accuracy (val):", flush=True)
        for key in sorted(cell_metrics.keys()):
            print(f"    {key}: {cell_metrics[key]*100:.1f}%", flush=True)

        eval_n = "4" if not use_tesseract else "6"
        print(f"\n[{eval_n}/7] Evaluating {mode} on {len(test_mazes)} test mazes...", flush=True)
        metrics = evaluate_model(model, test_mazes, DEVICE)
        print(f"  Success Rate:    {metrics['success_rate']:.2f}%", flush=True)
        print(f"  Optimal Rate:    {metrics['optimal_rate']:.2f}%", flush=True)
        print(f"  Avg Efficiency:  {metrics['avg_efficiency']:.2f}%", flush=True)
        print(flush=True)

        results[mode] = {"metrics": metrics, "cell_metrics": cell_metrics, "best_val_acc": best_val_acc}

        os.makedirs(os.path.join(os.path.dirname(__file__), "checkpoints"), exist_ok=True)
        ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"maze_{mode}_v5.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint: {ckpt_path}", flush=True)
        viz_ckpt_path = os.path.join(os.path.dirname(__file__), "viz_export", f"maze_{mode}_v5.pt")
        torch.save(model.state_dict(), viz_ckpt_path)
        print(f"  Checkpoint (viz_export copy): {viz_ckpt_path}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"  Tesseract v5 training complete. v4 checkpoint untouched.", flush=True)
    print(f"  Comparison against CE v3 runs separately (test_tesseract_v4_vs_ce.py).", flush=True)
    print(f"{'=' * 70}", flush=True)

    results_path = os.path.join(os.path.dirname(__file__), "maze_tesseract_v5_results.json")
    serializable = {}
    for mode, data in results.items():
        serializable[mode] = {
            "metrics": data["metrics"],
            "cell_metrics": {str(k): v for k, v in data["cell_metrics"].items()},
            "best_val_acc": data["best_val_acc"],
        }
    serializable["config"] = {
        "seed": SEED, "train_mazes": NUM_TRAIN_MAZES, "val_mazes": NUM_VAL_MAZES,
        "test_mazes": NUM_TEST_MAZES, "epochs": EPOCHS, "lr": LR,
        "lr_tesseract": LR_TESSERACT, "batch_size": BATCH_SIZE, "device": DEVICE,
        "training_mode": "all-position BFS",
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results: {results_path}", flush=True)


if __name__ == "__main__":
    run_experiment()
