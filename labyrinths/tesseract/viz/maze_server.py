"""
Labyrinth Solver API Server
Loads CE baseline and Tesseract checkpoints, generates mazes, runs inference.
"""

import os
import sys
import json
import time
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---- constants (must match training script) ----
GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE
VOCAB_SIZE = 10
WALKABLE_TOKENS = {0, 1, 2}
MAX_EVAL_STEPS = 50
REVISIT_PENALTY = 0.01

# ---- model architecture (exact copy from training) ----
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

# ---- inference ----
def flatten_grid(grid_2d):
    return [grid_2d[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]

def solve_autoregressive(model, grid_2d, start, end, device, max_steps=MAX_EVAL_STEPS,
                         no_revisit=True):
    """When no_revisit=True: prefer unvisited neighbors (model-ranked), LRU fallback.
    When no_revisit=False: original logic with soft revisit penalty (0.01)."""
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

            if no_revisit:
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
            else:
                best_prob, best_pos = -1.0, None
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if grid_2d[nr][nc] in WALKABLE_TOKENS:
                            p = probs[nr * GRID_SIZE + nc].item()
                            if (nr, nc) in visited:
                                p *= REVISIT_PENALTY
                            if p > best_prob:
                                best_prob, best_pos = p, (nr, nc)
                if best_pos is None:
                    break
                curr_r, curr_c = best_pos

            path.append((curr_r, curr_c))
            visited.add((curr_r, curr_c))
            visit_order[(curr_r, curr_c)] = step
    return path

FOG_TOKEN = 9  # unknown/unseen cell token (within VOCAB_SIZE=10)
FOG_VISION = 2  # radius: reveals a 5x5 square around agent

def solve_fog_of_war(model, grid_2d, start, end, device, max_steps=MAX_EVAL_STEPS,
                     no_revisit=True, vision=FOG_VISION):
    """Fog-of-war inference: model only sees cells within vision radius of
    positions it has visited. Unseen cells are replaced with FOG_TOKEN.
    Returns (path, list_of_revealed_sets) for frontend visualization."""
    model.eval()
    revealed = set()
    curr_r, curr_c = start
    path = [(curr_r, curr_c)]
    visited = {(curr_r, curr_c)}
    visit_order = {(curr_r, curr_c): 0}
    step = 0
    reveals_per_step = []

    def reveal_around(r, c):
        for dr in range(-vision, vision + 1):
            for dc in range(-vision, vision + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    revealed.add((nr, nc))

    # Reveal initial position
    reveal_around(curr_r, curr_c)
    reveals_per_step.append(sorted(revealed))

    with torch.no_grad():
        for _ in range(max_steps):
            if (curr_r, curr_c) == end:
                break
            step += 1

            # Build fog grid: only revealed cells show true values
            fog_flat = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) in revealed:
                        fog_flat.append(grid_2d[r][c])
                    else:
                        fog_flat.append(FOG_TOKEN)

            grid_t = torch.tensor(fog_flat, dtype=torch.long, device=device).unsqueeze(0)
            curr_idx = curr_r * GRID_SIZE + curr_c
            curr_t = torch.tensor(curr_idx, dtype=torch.long, device=device).unsqueeze(0)
            probs = F.softmax(model(grid_t, curr_t), dim=-1).squeeze(0)

            if no_revisit:
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
            else:
                best_prob, best_pos = -1.0, None
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if grid_2d[nr][nc] in WALKABLE_TOKENS:
                            p = probs[nr * GRID_SIZE + nc].item()
                            if (nr, nc) in visited:
                                p *= REVISIT_PENALTY
                            if p > best_prob:
                                best_prob, best_pos = p, (nr, nc)
                if best_pos is None:
                    break
                curr_r, curr_c = best_pos

            path.append((curr_r, curr_c))
            visited.add((curr_r, curr_c))
            visit_order[(curr_r, curr_c)] = step
            reveal_around(curr_r, curr_c)
            reveals_per_step.append(sorted(revealed))

    return path, reveals_per_step


# ---- external maze generator ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import generate_labyrinth, solve_bfs

# ---- external checkpoint architectures (compare_architectures.py) ----
# Named External* to avoid colliding with our own LabyrinthTransformer (embed_dim=64) above.
# State dict keys must match exactly: grid_embedding, pos_embedding (solver only),
# spatial_embedding, transformer.*, fc_out.weight/fc_out.bias (plain nn.Linear).
class ExternalReconstructor(nn.Module):
    """Reconstructs a full 10x10 grid from a partially visible one.
    Input (batch,100) tokens 0-9. Output (batch,100,10) per-cell class logits."""
    def __init__(self, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2):
        super().__init__()
        self.grid_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.spatial_embedding = nn.Parameter(torch.randn(1, NUM_CELLS, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, VOCAB_SIZE)

    def forward(self, grid):
        x = self.grid_embedding(grid) + self.spatial_embedding
        out = self.transformer(x)
        return self.fc_out(out)


class ExternalSolver(nn.Module):
    """Predicts the next navigation step. Used for both monolithic_solver.pt
    (fed partial grids) and modular_solver.pt (fed reconstructed grids)."""
    def __init__(self, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2):
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

# ---- External inference (compare_architectures.py: solve_autoregressive_monolithic /
# solve_autoregressive_modular / get_partial_visibility_grid) ----
EXTERNAL_VISION = 1  # native vision radius: 3x3 box (Chebyshev distance 1)

def external_box_reveal(positions, vision=EXTERNAL_VISION):
    """Box-reveal (Chebyshev distance) around a set of positions."""
    revealed = set()
    for (r, c) in positions:
        for dr in range(-vision, vision + 1):
            for dc in range(-vision, vision + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    revealed.add((nr, nc))
    return revealed

def external_partial_grid(flat_grid, visited_positions, start, end, vision=EXTERNAL_VISION):
    """Matches get_partial_visibility_grid exactly: box-reveal around every
    visited position cumulatively, plus start/end always revealed. Unseen = FOG_TOKEN."""
    partial = [FOG_TOKEN] * NUM_CELLS
    start_idx = start[0] * GRID_SIZE + start[1]
    end_idx = end[0] * GRID_SIZE + end[1]
    partial[start_idx] = flat_grid[start_idx]
    partial[end_idx] = flat_grid[end_idx]
    revealed = external_box_reveal(visited_positions, vision)
    for (r, c) in revealed:
        idx = r * GRID_SIZE + c
        partial[idx] = flat_grid[idx]
    revealed.add(start)
    revealed.add(end)
    return partial, revealed

def _build_external_test_grid(grid_2d, start, end):
    """Force start/end cells to external token convention (1=start, 2=end),
    matching the grid_test construction in solve_autoregressive_monolithic/modular."""
    grid_test = [row[:] for row in grid_2d]
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid_test[r][c] in (1, 2):
                grid_test[r][c] = 0
    grid_test[start[0]][start[1]] = 1
    grid_test[end[0]][end[1]] = 2
    return grid_test

def _external_pick_neighbor(probs, grid_test, curr_pos, visited):
    """Soft revisit penalty only (0.01), no hard block - matches external code."""
    r, c = curr_pos
    best_neighbor, best_prob = None, -1.0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            if grid_test[nr][nc] in (0, 1, 2):
                p = probs[nr * GRID_SIZE + nc].item()
                if (nr, nc) in visited:
                    p *= REVISIT_PENALTY
                if p > best_prob:
                    best_prob, best_neighbor = p, (nr, nc)
    return best_neighbor

def solve_external_monolithic(model, grid_2d, start, end, device, fog=True,
                              max_steps=MAX_EVAL_STEPS, vision=EXTERNAL_VISION):
    """Replicates solve_autoregressive_monolithic. When fog=True uses external
    native partial visibility (vision=1); when fog=False the raw full grid is fed
    directly (zero-shot, he never trained/tested this way)."""
    model.eval()
    grid_test = _build_external_test_grid(grid_2d, start, end)
    flat_test_grid = flatten_grid(grid_test)

    visited = {start}
    path = [start]
    curr_pos = start
    reveals_per_step = None

    if fog:
        revealed0 = external_box_reveal([start], vision)
        revealed0.add(start); revealed0.add(end)
        reveals_per_step = [sorted(revealed0)]

    with torch.no_grad():
        for _ in range(max_steps):
            if curr_pos == end:
                break

            if fog:
                g_partial, revealed = external_partial_grid(flat_test_grid, path, start, end, vision)
            else:
                g_partial = flat_test_grid

            g_partial_t = torch.tensor([g_partial], dtype=torch.long, device=device)
            curr_idx = curr_pos[0] * GRID_SIZE + curr_pos[1]
            curr_t = torch.tensor([curr_idx], dtype=torch.long, device=device)
            probs = F.softmax(model(g_partial_t, curr_t), dim=-1).squeeze(0)

            best_neighbor = _external_pick_neighbor(probs, grid_test, curr_pos, visited)
            if best_neighbor is None:
                break
            curr_pos = best_neighbor
            path.append(curr_pos)
            visited.add(curr_pos)
            if fog:
                reveals_per_step.append(sorted(revealed))

    return path, reveals_per_step

def solve_external_modular(recon_model, solver_model, grid_2d, start, end, device, fog=True,
                           max_steps=MAX_EVAL_STEPS, vision=EXTERNAL_VISION):
    """Replicates solve_autoregressive_modular: reconstruct the full grid from
    partial visibility (or raw full grid if fog=False), then run the modular
    solver on the reconstruction. Known walkable/start/end cells (from ground
    truth) are forced back onto the reconstruction each step, exactly as in
    compare_architectures.py."""
    recon_model.eval()
    solver_model.eval()
    grid_test = _build_external_test_grid(grid_2d, start, end)
    flat_test_grid = flatten_grid(grid_test)

    visited = {start}
    path = [start]
    curr_pos = start
    reveals_per_step = None

    if fog:
        revealed0 = external_box_reveal([start], vision)
        revealed0.add(start); revealed0.add(end)
        reveals_per_step = [sorted(revealed0)]

    with torch.no_grad():
        for _ in range(max_steps):
            if curr_pos == end:
                break

            if fog:
                g_partial, revealed = external_partial_grid(flat_test_grid, path, start, end, vision)
            else:
                g_partial = flat_test_grid

            g_partial_t = torch.tensor([g_partial], dtype=torch.long, device=device)
            recon_logits = recon_model(g_partial_t)
            recon_grid = torch.argmax(recon_logits, dim=-1)

            for idx in range(NUM_CELLS):
                if flat_test_grid[idx] in (0, 1, 2):
                    recon_grid[0, idx] = 0
            recon_grid[0, start[0] * GRID_SIZE + start[1]] = 1
            recon_grid[0, end[0] * GRID_SIZE + end[1]] = 2

            curr_idx = curr_pos[0] * GRID_SIZE + curr_pos[1]
            curr_t = torch.tensor([curr_idx], dtype=torch.long, device=device)
            probs = F.softmax(solver_model(recon_grid, curr_t), dim=-1).squeeze(0)

            best_neighbor = _external_pick_neighbor(probs, grid_test, curr_pos, visited)
            if best_neighbor is None:
                break
            curr_pos = best_neighbor
            path.append(curr_pos)
            visited.add(curr_pos)
            if fog:
                reveals_per_step.append(sorted(revealed))

    return path, reveals_per_step

# ---- server config ----
PORT = 8765
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), "Intro-NN", "labs")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maze_session_log.json")

# ---- difficulty presets ----
# corner_offset: how far from the corner edge (0=right in the corner, 2=up to 2 cells in)
DIFFICULTY = {
    "easy":    {"loops": (1, 2),  "dead_ends": (1, 2), "corner_offset": (0, 2)},
    "medium":  {"loops": (4, 6),  "dead_ends": (2, 3), "corner_offset": (0, 2)},
    "hard":    {"loops": (7, 9),  "dead_ends": (3, 5), "corner_offset": (0, 3)},
    "extreme": {"loops": (10, 15), "dead_ends": (5, 8), "corner_offset": (0, 1)},
}

# ---- load models ----
def load_model(checkpoint_path):
    model = LabyrinthTransformer()
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_external_model(model_cls, checkpoint_path):
    model = model_cls()
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

ce_model = None
tess_model = None
external_mono_model = None
external_modular_model = None
external_recon_model = None
PRIMARY_MODELS = {}

MODEL_NOTE_ZERO_SHOT = ("external models were trained only on partial visibility (3x3) - "
                        "full visibility here is zero-shot.")

def init_models():
    global ce_model, tess_model, PRIMARY_MODELS
    ce_path = os.path.join(CHECKPOINT_DIR, "maze_baseline_ce_v3.pt")
    tess_path = os.path.join(CHECKPOINT_DIR, "maze_tesseract_v5.pt")
    print(f"Loading CE model from {ce_path} ...")
    ce_model = load_model(ce_path)
    print(f"Loading Tesseract model from {tess_path} ...")
    tess_model = load_model(tess_path)
    PRIMARY_MODELS = {"baseline_ce": ce_model, "primary_tesseract": tess_model}
    print(f"Primary models loaded on {DEVICE}.")

def init_external_models():
    global external_mono_model, external_modular_model, external_recon_model
    mono_path = os.path.join(EXTERNAL_DIR, "monolithic_solver.pt")
    mod_path = os.path.join(EXTERNAL_DIR, "modular_solver.pt")
    recon_path = os.path.join(EXTERNAL_DIR, "reconstructor.pt")
    print(f"Loading External monolithic solver from {mono_path} ...")
    external_mono_model = load_external_model(ExternalSolver, mono_path)
    print(f"Loading External modular solver from {mod_path} ...")
    external_modular_model = load_external_model(ExternalSolver, mod_path)
    print(f"Loading External reconstructor from {recon_path} ...")
    external_recon_model = load_external_model(ExternalReconstructor, recon_path)
    print(f"External models loaded on {DEVICE}.")

def run_model(model_key, grid, start, end, no_revisit, fog):
    """Dispatch to the right solver for a model key. Returns (path, reveals, note)."""
    if model_key in PRIMARY_MODELS:
        model_obj = PRIMARY_MODELS[model_key]
        if fog:
            path, reveals = solve_fog_of_war(model_obj, grid, start, end, DEVICE, no_revisit=no_revisit)
        else:
            path = solve_autoregressive(model_obj, grid, start, end, DEVICE, no_revisit=no_revisit)
            reveals = None
        return path, reveals, ""
    elif model_key == "external_monolithic":
        path, reveals = solve_external_monolithic(external_mono_model, grid, start, end, DEVICE, fog=fog)
        return path, reveals, ("" if fog else MODEL_NOTE_ZERO_SHOT)
    elif model_key == "external_modular":
        path, reveals = solve_external_modular(external_recon_model, external_modular_model,
                                               grid, start, end, DEVICE, fog=fog)
        return path, reveals, ("" if fog else MODEL_NOTE_ZERO_SHOT)
    else:
        raise ValueError(f"Unknown model: {model_key}")

# ---- session log ----
log_lock = threading.Lock()

def read_log():
    if not os.path.exists(LOG_FILE):
        return {"sessions": []}
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"sessions": []}

def write_log(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---- maze generation ----
# Direction templates: start corner -> end corner
# Model was trained on top-left -> bottom-right; mixing directions tests generalization
DIRECTIONS = [
    ("top-left",     "bottom-right"),
    ("bottom-right", "top-left"),
    ("top-right",    "bottom-left"),
    ("bottom-left",  "top-right"),
]

def pick_position(corner, offset_lo, offset_hi):
    """Pick (row, col) in a corner region of the grid.
    offset_lo/hi are distance from the corner edge (0 = right in the corner)."""
    if corner.startswith("top"):
        r = random.randint(offset_lo, offset_hi)
    else:
        r = random.randint(GRID_SIZE - 1 - offset_hi, GRID_SIZE - 1 - offset_lo)
    if corner.endswith("left"):
        c = random.randint(offset_lo, offset_hi)
    else:
        c = random.randint(GRID_SIZE - 1 - offset_hi, GRID_SIZE - 1 - offset_lo)
    return (r, c)

def generate_maze(difficulty):
    preset = DIFFICULTY.get(difficulty, DIFFICULTY["medium"])
    lo_loop, hi_loop = preset["loops"]
    off_lo, off_hi = preset["corner_offset"]

    for attempt in range(30):
        num_loops = random.randint(lo_loop, hi_loop)
        start_corner, end_corner = random.choice(DIRECTIONS)
        start = pick_position(start_corner, off_lo, off_hi)
        end = pick_position(end_corner, off_lo, off_hi)

        grid = generate_labyrinth(
            width=GRID_SIZE, height=GRID_SIZE,
            start=start, end=end,
            num_dead_ends=random.randint(*preset["dead_ends"]),
            num_loops=num_loops,
        )
        bfs_path = solve_bfs(grid, start, end)
        if bfs_path is not None and len(bfs_path) > 5:
            return grid, start, end, bfs_path

    return None, None, None, None

# ---- request handler ----
class MazeHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[{self.command}] {self.path} - {args[0] if args else ''}")

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html_bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors_headers()
        self.send_header("Content-Length", str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)

    def _error_response(self, msg, status=500):
        self._json_response({"error": msg}, status)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        try:
            if self.path == "/" or self.path == "/index.html":
                html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maze_viz.html")
                with open(html_path, "rb") as f:
                    self._html_response(f.read())
            elif self.path == "/api/log":
                with log_lock:
                    data = read_log()
                self._json_response(data)
            else:
                self._error_response("Not found", 404)
        except Exception as e:
            self._error_response(str(e))

    def do_POST(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            payload = json.loads(body) if body else {}

            if self.path == "/api/generate":
                difficulty = payload.get("difficulty", "medium").lower()
                if difficulty not in DIFFICULTY:
                    difficulty = "medium"

                no_revisit = payload.get("no_revisit", True)
                fog = payload.get("fog_of_war", False)

                valid_models = {"baseline_ce", "primary_tesseract", "external_monolithic", "external_modular"}
                model_left = payload.get("model_left", "baseline_ce")
                model_right = payload.get("model_right", "primary_tesseract")
                if model_left not in valid_models:
                    model_left = "baseline_ce"
                if model_right not in valid_models:
                    model_right = "primary_tesseract"

                grid, start, end, bfs_path = generate_maze(difficulty)
                if grid is None:
                    self._error_response("Failed to generate a solvable maze after 20 attempts", 500)
                    return

                left_path, left_reveals, left_note = run_model(
                    model_left, grid, start, end, no_revisit, fog)
                right_path, right_reveals, right_note = run_model(
                    model_right, grid, start, end, no_revisit, fog)

                left_reached = left_path[-1] == end
                right_reached = right_path[-1] == end

                result = {
                    "grid": grid,
                    "start": list(start),
                    "end": list(end),
                    "optimal_path": [list(p) for p in bfs_path],
                    "optimal_length": len(bfs_path),
                    "difficulty": difficulty,
                    "fog_of_war": fog,
                    "no_revisit": no_revisit,
                    "model_left": model_left,
                    "model_right": model_right,
                    "left_path": [list(p) for p in left_path],
                    "right_path": [list(p) for p in right_path],
                    "left_reached_goal": left_reached,
                    "right_reached_goal": right_reached,
                    "left_note": left_note,
                    "right_note": right_note,
                }
                if left_reveals is not None:
                    result["left_reveals"] = [[[r, c] for r, c in s] for s in left_reveals]
                if right_reveals is not None:
                    result["right_reveals"] = [[[r, c] for r, c in s] for s in right_reveals]
                self._json_response(result)

            elif self.path == "/api/log":
                with log_lock:
                    data = read_log()
                    data["sessions"].append(payload)
                    write_log(data)
                self._json_response({"status": "ok"})

            else:
                self._error_response("Not found", 404)

        except Exception as e:
            self._error_response(str(e))

# ---- threaded server ----
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def main():
    init_models()
    init_external_models()
    server = ThreadedHTTPServer(("0.0.0.0", PORT), MazeHandler)
    print(f"Models loaded. Server running at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()

if __name__ == "__main__":
    main()
