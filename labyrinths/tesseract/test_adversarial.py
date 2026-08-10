"""
Adversarial maze test for Tesseract v5 vs CE baseline v3
==========================================================
Hand-crafted 10x10 mazes designed to break a "move toward the goal"
greedy heuristic: dead-end branches that point toward the goal, forced
detours away from the goal before the true path opens up, spirals,
zigzags, and bottlenecks. If a model is actually doing graph reasoning
(BFS-like), it should navigate these efficiently. If it's just greedily
moving toward the goal by Euclidean/Manhattan proximity, it should get
lured into dead ends and take much longer paths (or fail entirely).

Model architecture, solve_autoregressive, and token semantics are
copied verbatim from maze_tesseract_v5_cpu.py (which itself imports the
BFS solver / labyrinth generator from Intro-NN/labs/solver.py). We do
not import maze_tesseract_v5_cpu.py directly because it has heavy
argparse-free but still executable module-level training constants and
a __main__ guard; the safe approach requested by the task is to copy
the exact classes/functions we need.

Token semantics (confirmed from maze_tesseract_v5_cpu.py and
Intro-NN/labs/solver.py's generate_labyrinth docstring):
  0 = walkable path
  1 = start
  2 = end
  3-8 = visible wall (randomly chosen per-wall-cell)
  9 = hidden/non-visible wall (interior wall not adjacent to any
      walkable cell)
WALKABLE_TOKENS = {0, 1, 2} in the training code -- tokens 3-8 and 9
are ALL non-walkable for both the BFS solver and the model's own
training signal (bfs_all_next_steps in maze_tesseract_v5_cpu.py uses
this exact same WALKABLE_TOKENS set to decide adjacency). The model
does NOT receive any special "9 looks walkable" treatment -- it's just
a distinct embedding index (vocab id 9) that is, like ids 3-8, never a
BFS-adjacency target during training. So our BFS solver here treats
3-9 uniformly as walls, matching the model's own training-time ground
truth exactly.
"""

import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (copied verbatim from maze_tesseract_v5_cpu.py) ───────
GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE
VOCAB_SIZE = 10
WALKABLE_TOKENS = {0, 1, 2}
MAX_EVAL_STEPS = 50

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TESSERACT_CKPT = os.path.join(CKPT_DIR, "maze_tesseract_v5.pt")
CE_CKPT = os.path.join(CKPT_DIR, "maze_baseline_ce_v3.pt")
DEVICE = "cpu"


# ── Model (copied verbatim from maze_tesseract_v5_cpu.py) ───────────

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


def flatten_grid(grid_2d):
    return [grid_2d[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


# ── Autoregressive solver (copied verbatim from maze_tesseract_v5_cpu.py) ──

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
    hit_step_limit = False
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
        else:
            # loop completed without break -> ran out of steps
            hit_step_limit = (curr_r, curr_c) != end
    return path, hit_step_limit


# ── BFS shortest-path solver ─────────────────────────────────────────
# Matches WALKABLE_TOKENS exactly: walls are any token 3-9 (visible 3-8,
# hidden 9), all uniformly non-walkable -- same ground truth the model
# was trained against (bfs_all_next_steps in the training script uses
# this identical WALKABLE_TOKENS set).

def bfs_shortest_path(grid_2d, start, end):
    if grid_2d[start[0]][start[1]] not in WALKABLE_TOKENS:
        return None
    queue = deque([start])
    visited = {start}
    parent = {}
    while queue:
        cell = queue.popleft()
        if cell == end:
            path = [cell]
            while cell in parent:
                cell = parent[cell]
                path.append(cell)
            path.reverse()
            return path
        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if (nr, nc) not in visited and grid_2d[nr][nc] in WALKABLE_TOKENS:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = cell
                    queue.append((nr, nc))
    return None


# ── Maze validation helper ───────────────────────────────────────────

def validate_maze(name, grid_2d, expected_start, expected_end):
    flat = flatten_grid(grid_2d)
    ones = flat.count(1)
    twos = flat.count(2)
    assert ones == 1, f"{name}: expected exactly one '1' (start) cell, found {ones}"
    assert twos == 1, f"{name}: expected exactly one '2' (end) cell, found {twos}"
    start_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid_2d[r][c] == 1]
    end_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid_2d[r][c] == 2]
    assert start_cells[0] == expected_start, f"{name}: start cell {start_cells[0]} != expected {expected_start}"
    assert end_cells[0] == expected_end, f"{name}: end cell {end_cells[0]} != expected {expected_end}"
    path = bfs_shortest_path(grid_2d, expected_start, expected_end)
    assert path is not None, f"{name}: BFS found NO path from {expected_start} to {expected_end}"
    return path


# ── ASCII visualization ──────────────────────────────────────────────

def render_grid(grid_2d, start, end):
    lines = []
    for r in range(GRID_SIZE):
        row_chars = []
        for c in range(GRID_SIZE):
            if (r, c) == start:
                row_chars.append("S")
            elif (r, c) == end:
                row_chars.append("E")
            elif grid_2d[r][c] == 0:
                row_chars.append(".")
            else:
                row_chars.append("#")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def render_path_overlay(grid_2d, start, end, path, marker):
    path_set = set(path)
    lines = []
    for r in range(GRID_SIZE):
        row_chars = []
        for c in range(GRID_SIZE):
            if (r, c) == start:
                row_chars.append("S")
            elif (r, c) == end:
                row_chars.append("E")
            elif (r, c) in path_set:
                row_chars.append(marker)
            elif grid_2d[r][c] == 0:
                row_chars.append(".")
            else:
                row_chars.append("#")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def compact_path(path):
    return " -> ".join(f"({r},{c})" for r, c in path)


# ── Build the 6 adversarial mazes ────────────────────────────────────
# Grid values: 0=walkable interior, 1=start, 2=end, 3-8=visible wall
# (varied per maze), 9=hidden wall. All non-zero-non-1-non-2 values are
# walls (non-walkable), matching WALKABLE_TOKENS treatment exactly.

def build_maze_1_detour():
    """The Detour: start (0,0), end (9,9). A vertical wall at col 4 (open
    only at row 5) splits the grid into a left half and a right half, so
    the model must travel down to row 5 before it can cross to the right
    side at all. Once across, a further wall forces an excursion back up
    toward row 2 before the path can wrap down to (9,9) -- so the true
    shortest path is not monotonically toward the goal. Several dead-end
    branches are carved into both halves."""
    W = 3  # wall token used for this maze's visible walls
    g = [[0] * 10 for _ in range(10)]
    # Vertical wall at col4, open only at the row5 gap -> forces descent
    # to row5 before any left-to-right crossing is possible at all.
    for r in range(10):
        if r != 5:
            g[r][4] = W
    # A solid 3x3 block plus row/col seals in the right half forces the
    # crossing at (5,5) to detour UP to row3 (away from the goal, so the
    # BFS path is genuinely non-monotonic) before it can loop back down
    # through col9 to reach (9,9). The lower shortcut at col5 is sealed
    # off so the up-and-over detour is the only way through.
    for r in range(6, 9):
        for c in range(6, 9):
            g[r][c] = W
    for c in range(6, 9):
        g[5][c] = W
    for r in range(6, 10):
        g[r][5] = W
    # Dead end 1: pocket in the left half near (2,1)-(2,2)
    g[2][1] = 0
    g[2][2] = 0
    g[1][2] = W
    g[3][2] = W
    # Dead end 2: lure along row6 in the left half that looks like it
    # heads toward the goal but stops short
    g[6][1] = 0
    g[6][2] = 0
    g[7][2] = W
    g[6][3] = W
    # Dead end 3: pocket near (8,0)-(8,1) in the left half
    g[8][1] = 0
    g[8][0] = 0
    g[9][1] = W
    # Dead end 4: tempting nub in the right half near (2,6)-(2,7) close
    # to the true path but walled off from it
    g[2][6] = 0
    g[2][7] = 0
    g[1][6] = W
    g[1][7] = W
    g[3][6] = W
    g[3][7] = W
    g[2][8] = W
    # Dead end 5: small pocket right next to the goal (row9, cols6-8)
    # that looks very tempting (adjacent to E) but only connects to the
    # goal itself, not to the rest of the maze -- a lure, not a shortcut.
    g[9][6] = 0
    g[9][7] = 0
    g[9][8] = 0
    g[0][0] = 1
    g[9][9] = 2
    return g


def build_maze_2_spiral():
    """The Spiral: start (0,0), end (5,5) (near center). A genuine 1-cell-
    wide clockwise inward spiral corridor: right along row0 to col9, down
    col9 to row9, left along row9 to col0, up col0 to row2, right along
    row2 toward the center, down, left, up again for a tighter inner
    ring, then in to (5,5). Built as an explicit corridor coordinate list
    (not full open "ring rectangles") so consecutive laps never touch --
    that is what forces the actual spiral traversal instead of letting a
    solver cut across a lap early."""
    W = 3
    g = [[W] * 10 for _ in range(10)]
    corridor = []
    corridor += [(0, c) for c in range(0, 10)]          # ring0 top: row0, col0->9
    corridor += [(r, 9) for r in range(1, 10)]           # ring0 right: col9, row1->9
    corridor += [(9, c) for c in range(8, -1, -1)]       # ring0 bottom: row9, col8->0
    corridor += [(r, 0) for r in range(8, 1, -1)]        # ring0 left: col0, row8->2
    corridor += [(2, c) for c in range(1, 8)]            # ring1 top: row2, col1->7
    corridor += [(r, 7) for r in range(3, 8)]            # ring1 right: col7, row3->7
    corridor += [(7, c) for c in range(6, 1, -1)]        # ring1 bottom: row7, col6->2
    corridor += [(r, 2) for r in range(6, 3, -1)]        # ring1 left: col2, row6->4
    corridor += [(4, c) for c in range(3, 6)]            # ring2: row4, col3->5
    corridor += [(5, 5)]                                 # in to the goal
    for r, c in corridor:
        g[r][c] = 0
    # One genuine dead-end nub: the tightly-wound single-width spiral
    # leaves almost no spare buffer space (any wider nub would touch two
    # laps at once and become an accidental shortcut instead of a dead
    # end), but (6,4) is a verified true dead end -- its only open
    # neighbor is the ring2 row4 corridor, and it goes nowhere else.
    g[6][4] = 0
    g[0][0] = 1
    g[5][5] = 2
    return g


def build_maze_3_false_shortcut():
    """The False Shortcut: start (0,0), end (9,9). A tempting diagonal
    corridor heads straight toward the goal and dead-ends around (5,5).
    The real path must first go further along row/col away from the
    goal direction before wrapping around."""
    W = 4
    g = [[W] * 10 for _ in range(10)]
    # The lure: diagonal-ish corridor from (0,0) toward (5,5) that dead-ends
    lure = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5)]
    for r, c in lure:
        g[r][c] = 0
    # dead end nubs off the lure
    g[1][2] = 0
    g[0][1] = 0  # small pocket off start
    g[3][4] = 0
    g[3][5] = 0  # pokes toward goal then stops (neighbors are wall)
    # The real path: go along row 0 further right (away/orthogonal), then
    # down col 9, then left along a corridor to reach (9,9) wrapping the
    # bottom, bypassing the lure's dead end entirely.
    for c in range(0, 10):
        g[0][c] = 0
    for r in range(0, 10):
        g[r][9] = 0
    for c in range(6, 10):
        g[8][c] = 0
    for r in range(6, 9):
        g[r][6] = 0
    for c in range(6, 10):
        g[9][c] = 0
    # extra dead ends near the wrap
    g[7][7] = 0
    g[7][8] = W  # nub that stops
    g[2][9] = 0  # already part of col9 corridor; fine
    g[1][8] = 0
    g[1][7] = W  # small dead end off col9 corridor
    g[0][0] = 1
    g[9][9] = 2
    return g


def build_maze_4_zigzag():
    """The Zigzag: start (0,0), end (9,0). A boustrophedon corridor: full
    horizontal rows 0, 2, 4, 6 alternate direction (right along row0 to
    col9, left along row2 to col0, right along row4 to col9, left along
    row6 to col0), each pair connected by a single-cell gap in the thin
    "connector" row between them (rows 1, 3, 5). This is the standard way
    to force a true zigzag in a 4-connected grid: if every row were fully
    open, row-to-row adjacency would exist at every column and the model
    could just shoot straight down col 0 in 9 steps -- so only rows 0/2/4/6
    are full corridors, and rows 1/3/5 are wall except the single turn
    gap. After row6 (ending back at col0), a narrow single-column
    approach through rows 7-9 (open only at col0) leads to the goal --
    which conveniently frees up the rest of rows 7-8 as genuine dead-end
    pockets hanging off row6/row7 that go nowhere (row8 stays walled at
    every other column, so these pockets can't act as shortcuts)."""
    W = 5
    g = [[W] * 10 for _ in range(10)]
    corridor_rows = [0, 2, 4, 6]
    turn_after = {}
    direction = 1  # 1 = left->right (ends at col9), -1 = right->left (ends at col0)
    for r in corridor_rows:
        for c in range(10):
            g[r][c] = 0
        turn_after[r] = 9 if direction == 1 else 0
        direction *= -1
    for r in [1, 3, 5]:
        g[r][turn_after[r - 1]] = 0

    # Narrow final approach: connector row7 opens at the same column row6
    # ended on (col0), then rows 7-9 stay open only at col0 down to the goal.
    final_col = turn_after[6]
    g[7][final_col] = 0
    for r in [7, 8, 9]:
        g[r][0] = 0

    # Genuine dead-end pockets off row7 -- each is a single open cell that
    # only touches row6 (fully open) above it; row8 is wall at every
    # column except col0, so none of these lead anywhere.
    for c in [2, 4, 6, 8]:
        g[7][c] = 0
    g[8][4] = 0  # one pocket extended a cell deeper, still a dead end

    g[0][0] = 1
    g[9][0] = 2
    return g


def build_maze_5_bottleneck():
    """The Bottleneck: start (0,0), end (9,9). Two open regions
    connected only through a single-cell chokepoint around (5,0), far
    from the direct diagonal. Dead ends carved in both regions."""
    W = 6
    g = [[0] * 10 for _ in range(10)]
    # Wall off a band splitting the grid into a "top" region (rows 0-4)
    # and "bottom" region (rows 5-9), except a single chokepoint at col0
    # (row4/col0 and row5/col0 both stay open, everything else in that
    # band is wall) -- so the only way from top to bottom is through
    # (4,0) -> (5,0), far from the direct diagonal toward (9,9).
    for c in range(1, 10):
        g[4][c] = W
    for c in range(1, 10):
        g[5][c] = W
    # Dead ends in top region (rows 0-3)
    g[1][5] = W
    g[1][6] = 0
    g[1][7] = 0
    g[0][7] = W
    g[2][7] = W
    g[1][8] = W  # seals dead-end nub at (1,6)-(1,7)
    g[3][2] = 0
    g[3][3] = 0
    g[2][3] = W
    g[4][3] = W  # already wall (band) but explicit
    g[3][4] = W  # seal nub
    # Dead ends in bottom region (rows 6-9)
    g[7][2] = 0
    g[7][3] = 0
    g[6][3] = W
    g[8][3] = W
    g[7][4] = W  # seal nub
    g[8][7] = 0
    g[9][7] = 0
    g[9][6] = W
    g[8][6] = W  # seal nub
    g[6][8] = 0
    g[5][8] = W
    g[7][8] = W  # seal nub near goal, tempting but isolated
    g[0][0] = 1
    g[9][9] = 2
    return g


def build_maze_6_uturn():
    """The U-Turn: start (0,0), end (0,9). Only path: straight down col
    0 to row9, right along row9 to col9, up col9 to row0. Dead-end
    branches punched off the U corridor."""
    W = 8
    g = [[W] * 10 for _ in range(10)]
    for r in range(0, 10):
        g[r][0] = 0
    for c in range(0, 10):
        g[9][c] = 0
    for r in range(0, 10):
        g[r][9] = 0
    # Dead-end branches off the U corridor
    g[2][1] = 0
    g[2][2] = 0
    g[1][2] = W  # seal
    g[3][2] = W  # seal (nub stops at (2,2))
    g[5][1] = 0
    g[6][1] = 0
    g[6][2] = 0
    g[5][2] = W
    g[7][2] = W  # seal nub near col0 lower section
    g[9][3] = 0  # already part of row9
    g[8][3] = 0
    g[8][4] = 0
    g[7][3] = W
    g[7][4] = W  # seal nub off row9 corridor, tempts upward shortcut
    g[8][7] = 0
    g[7][7] = 0
    g[7][8] = W
    g[6][7] = W  # seal nub off col9 corridor near goal
    g[2][8] = 0
    g[3][8] = 0
    g[3][7] = W
    g[1][7] = W  # seal nub off col9 corridor
    g[0][0] = 1
    g[0][9] = 2
    return g


MAZES = [
    ("Maze 1: The Detour", build_maze_1_detour, (0, 0), (9, 9)),
    ("Maze 2: The Spiral", build_maze_2_spiral, (0, 0), (5, 5)),
    ("Maze 3: The False Shortcut", build_maze_3_false_shortcut, (0, 0), (9, 9)),
    ("Maze 4: The Zigzag", build_maze_4_zigzag, (0, 0), (9, 0)),
    ("Maze 5: The Bottleneck", build_maze_5_bottleneck, (0, 0), (9, 9)),
    ("Maze 6: The U-Turn", build_maze_6_uturn, (0, 0), (0, 9)),
]


def load_model(ckpt_path):
    model = LabyrinthTransformer()
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def has_revisits(path):
    return len(path) != len(set(path))


def main():
    print("=" * 78)
    print("ADVERSARIAL MAZE TEST: Tesseract v5 vs CE Baseline v3")
    print("=" * 78)

    # ── Build + validate all 6 mazes ──
    built = []
    for name, builder, start, end in MAZES:
        grid = builder()
        bfs_path = validate_maze(name, grid, start, end)
        built.append((name, grid, start, end, bfs_path))

    # Extra sanity checks per the task's intended-detour properties:
    # mazes 1,2,3,5,6 should NOT be a straight/monotonic path toward goal;
    # maze 4 should be long/zigzagging.
    def is_monotonic_toward_goal(path, start, end):
        # "monotonic" here: every step's Manhattan distance to goal
        # strictly decreases (i.e. a pure greedy-toward-goal path with no
        # backtracking or sideways detour).
        def mdist(p):
            return abs(p[0] - end[0]) + abs(p[1] - end[1])
        dists = [mdist(p) for p in path]
        return all(dists[i] > dists[i + 1] for i in range(len(dists) - 1))

    for name, grid, start, end, bfs_path in built:
        mono = is_monotonic_toward_goal(bfs_path, start, end)
        tag = "OK (forced detour present)" if not mono else "WARNING: BFS path is monotonic toward goal!"
        if "Zigzag" in name:
            tag = f"OK (zigzag length={len(bfs_path)})"
        print(f"[build-check] {name}: BFS optimal length={len(bfs_path)}  {tag}")
    print()

    # ── Load both models once ──
    print(f"Loading Tesseract v5 checkpoint: {TESSERACT_CKPT}")
    tesseract_model = load_model(TESSERACT_CKPT)
    print(f"Loading CE baseline v3 checkpoint: {CE_CKPT}")
    ce_model = load_model(CE_CKPT)
    print()

    summary_rows = []  # (maze_name, model_name, reached, path_len, ratio, revisited, hit_limit)

    for name, grid, start, end, bfs_path in built:
        print("=" * 78)
        print(name)
        print("=" * 78)
        print(f"Start: {start}   End: {end}")
        print(f"BFS optimal path length: {len(bfs_path)} cells ({len(bfs_path) - 1} steps)")
        print(f"BFS optimal path: {compact_path(bfs_path)}")
        print()
        print("Grid:")
        print(render_grid(grid, start, end))
        print()
        print("Grid with BFS optimal path overlay ('o'):")
        print(render_path_overlay(grid, start, end, bfs_path, "o"))
        print()

        for model_name, model in [("Tesseract v5", tesseract_model), ("CE Baseline v3", ce_model)]:
            path, hit_limit = solve_autoregressive(model, grid, start, end, DEVICE)
            reached = path[-1] == end
            revisited = has_revisits(path)
            ratio = (len(path) / len(bfs_path)) if reached else float("inf")
            print(f"--- {model_name} ---")
            print(f"  Reached goal: {'Y' if reached else 'N'}")
            print(f"  Path length taken: {len(path)} cells ({len(path) - 1} steps)")
            if reached:
                print(f"  Ratio to BFS optimal (path length / optimal length): {ratio:.2f}x")
            else:
                print(f"  Ratio to BFS optimal: N/A (did not reach goal)")
            print(f"  Revisited cells / went in circles: {'Y' if revisited else 'N'}")
            print(f"  Hit MAX_EVAL_STEPS ({MAX_EVAL_STEPS}) step limit without reaching goal: {'Y' if hit_limit else 'N'}")
            print(f"  Path taken: {compact_path(path)}")
            marker = "*"
            print(f"  Grid with {model_name} path overlay ('{marker}'):")
            print(render_path_overlay(grid, start, end, path, marker))
            print()
            summary_rows.append((name, model_name, reached, len(path), ratio, revisited, hit_limit))

    # ── Final summary table ──
    print("=" * 78)
    print("FINAL SUMMARY: Tesseract v5 vs CE Baseline v3 across 6 adversarial mazes")
    print("=" * 78)
    header = f"{'Maze':<28}{'Model':<16}{'Reached':<9}{'PathLen':<9}{'Ratio':<10}{'Revisit':<9}{'StepLimit':<10}"
    print(header)
    print("-" * len(header))
    for name, model_name, reached, path_len, ratio, revisited, hit_limit in summary_rows:
        ratio_str = f"{ratio:.2f}x" if ratio != float("inf") else "N/A"
        print(
            f"{name:<28}{model_name:<16}{('Y' if reached else 'N'):<9}"
            f"{path_len:<9}{ratio_str:<10}{('Y' if revisited else 'N'):<9}{('Y' if hit_limit else 'N'):<10}"
        )
    print()
    print("Done.")


if __name__ == "__main__":
    main()
