#!/usr/bin/env python3
"""
autonomous_map_agent.py

Autonomous MAP agent pipeline:
- Load a top-down map image.
- User clicks Start (A) and Goal (B).
- Agent auto-tunes segmentation parameters (threshold/polarity/morph/inflate)
  using task-aware validation:
    1) Start & Goal are free
    2) Start & Goal connected in free space
    3) A* path exists
    4) Prefer better clearance (distance from obstacles) + smoother maps

Visualization:
- Toggle raw vs occupancy overlay
- Draw start/goal + planned path
- Save occupancy + params

Controls:
- Left click: set Start (A), then Goal (B)
- C: clear Start/Goal
- SPACE: toggle raw / occupancy overlay
- R: rerun agent (after changing candidates or if you moved points)
- S: save outputs (occ.png + params.json)
- ESC: quit

Run:
  pip install pygame numpy
  python autonomous_map_agent.py "/absolute/path/to/map.png"
"""

import os
import sys
import json
import math
import heapq
from dataclasses import dataclass, asdict
from collections import deque

import pygame
import numpy as np


# -------------------------
# Map processing utilities
# -------------------------
class Map2D:
    """
    Image -> occupancy grid (bool): occ[y,x]=True means obstacle.

    Pipeline:
      - grayscale conversion
      - thresholding
      - morphology cleanup (closing-ish)
      - inflation (safety margin)
    """

    def __init__(
        self,
        map_surface: pygame.Surface,
        obstacle_is_dark: bool = True,
        morph_passes: int = 2,
        inflate_px: int = 10,
        threshold: float | None = None,
    ):
        self.surface = map_surface
        self.width = map_surface.get_width()
        self.height = map_surface.get_height()

        gray = self.surface_to_gray(map_surface)  # (H,W) float
        t = float(self.otsu_threshold(gray)) if threshold is None else float(threshold)

        occ = (gray < t) if obstacle_is_dark else (gray > t)

        if morph_passes > 0:
            occ = self.cleanup_static(occ, passes=morph_passes)

        if inflate_px > 0:
            occ = self.inflate_static(occ, radius=inflate_px)

        self.occ = occ.astype(bool)

    @staticmethod
    def surface_to_gray(surface: pygame.Surface) -> np.ndarray:
        arr = pygame.surfarray.array3d(surface).astype(np.float32)  # (W,H,3)
        arr = np.transpose(arr, (1, 0, 2))  # (H,W,3)
        gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        return gray

    @staticmethod
    def otsu_threshold(gray: np.ndarray) -> float:
        g = np.clip(gray, 0, 255).astype(np.uint8)
        hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
        total = g.size
        if total == 0:
            return 128.0

        p = hist / total
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]

        denom = omega * (1.0 - omega)
        denom[denom == 0] = 1e-12
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        return float(np.argmax(sigma_b2))

    @staticmethod
    def dilate8(mask: np.ndarray) -> np.ndarray:
        m = mask
        d = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                d |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        return d

    @staticmethod
    def erode8(mask: np.ndarray) -> np.ndarray:
        m = mask
        e = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                e &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        return e

    @staticmethod
    def cleanup_static(mask: np.ndarray, passes: int = 1) -> np.ndarray:
        m = mask.copy()
        for _ in range(passes):
            m = Map2D.dilate8(m)
        for _ in range(passes):
            m = Map2D.erode8(m)
        return m

    @staticmethod
    def inflate_static(occ: np.ndarray, radius: int) -> np.ndarray:
        m = occ.copy()
        for _ in range(int(radius)):
            m = Map2D.dilate8(m)
        return m


def occ_to_surface(occ: np.ndarray) -> pygame.Surface:
    """bool occ[y,x] -> surface (obstacle black, free white)."""
    h, w = occ.shape
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[occ] = (0, 0, 0)
    img_wh = np.transpose(img, (1, 0, 2))
    return pygame.surfarray.make_surface(img_wh)


def overlay_occ_on_map(map_surf: pygame.Surface, occ: np.ndarray, alpha=120) -> pygame.Surface:
    """Create an overlay: obstacles tinted dark on top of the map."""
    base = map_surf.copy()
    occ_s = occ_to_surface(occ).convert()
    occ_s.set_alpha(alpha)
    base.blit(occ_s, (0, 0))
    return base


# -------------------------
# Planning utilities (coarse grid)
# -------------------------
def downsample_occ(occ: np.ndarray, cell: int) -> np.ndarray:
    """
    Downsample pixel occupancy to coarse grid occupancy.
    A coarse cell is obstacle if ANY pixel in its block is obstacle.
    """
    H, W = occ.shape
    gh = (H + cell - 1) // cell
    gw = (W + cell - 1) // cell
    g = np.zeros((gh, gw), dtype=bool)

    for y in range(gh):
        y0 = y * cell
        y1 = min(H, y0 + cell)
        for x in range(gw):
            x0 = x * cell
            x1 = min(W, x0 + cell)
            g[y, x] = bool(np.any(occ[y0:y1, x0:x1]))
    return g


def pix_to_cell(pxy, cell: int):
    x, y = pxy
    return (int(x) // cell, int(y) // cell)  # (cx, cy)


def cell_to_pix_center(cxy, cell: int):
    cx, cy = cxy
    return (cx * cell + cell * 0.5, cy * cell + cell * 0.5)


def in_bounds(grid: np.ndarray, cy: int, cx: int) -> bool:
    return 0 <= cy < grid.shape[0] and 0 <= cx < grid.shape[1]


def bfs_connected(free: np.ndarray, start, goal) -> bool:
    """Fast connectivity check on coarse free grid (True=free)."""
    sx, sy = start
    gx, gy = goal
    if not in_bounds(free, sy, sx) or not in_bounds(free, gy, gx):
        return False
    if not free[sy, sx] or not free[gy, gx]:
        return False

    q = deque()
    q.append((sy, sx))
    seen = np.zeros_like(free, dtype=bool)
    seen[sy, sx] = True

    while q:
        y, x = q.popleft()
        if (x, y) == (gx, gy):
            return True
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = y + dy, x + dx
            if in_bounds(free, ny, nx) and free[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return False


def distance_to_obstacles(grid_occ: np.ndarray) -> np.ndarray:
    """
    Multi-source BFS distance (Manhattan) on coarse grid.
    Returns dist[y,x] = steps to nearest obstacle. Obstacles have dist=0.
    """
    h, w = grid_occ.shape
    dist = np.full((h, w), 1e9, dtype=np.float32)
    q = deque()

    ys, xs = np.where(grid_occ)
    for y, x in zip(ys, xs):
        dist[y, x] = 0.0
        q.append((y, x))

    # if no obstacles, return large distances
    if not q:
        dist[:] = 1e3
        return dist

    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] > d + 1:
                dist[ny, nx] = d + 1
                q.append((ny, nx))
    return dist


def astar_path(grid_occ: np.ndarray, start, goal):
    """
    A* on coarse grid with 8-neighbors.
    grid_occ[y,x]=True obstacle.
    start, goal are (cx,cy).
    Returns list of (cx,cy) or None.
    """
    sx, sy = start
    gx, gy = goal
    h, w = grid_occ.shape

    def free(cx, cy):
        return 0 <= cx < w and 0 <= cy < h and not grid_occ[cy, cx]

    if not free(sx, sy) or not free(gx, gy):
        return None

    def heuristic(a, b):
        # Euclidean
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # 8-neighbor moves
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    step_cost = {(dx,dy): (math.sqrt(2) if dx != 0 and dy != 0 else 1.0) for dx,dy in nbrs}

    open_heap = []
    heapq.heappush(open_heap, (0.0, (sx, sy)))
    came = {}
    gscore = {(sx, sy): 0.0}

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == (gx, gy):
            # reconstruct
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path

        for dx, dy in nbrs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not free(nx, ny):
                continue
            tentative = gscore[cur] + step_cost[(dx,dy)]
            if (nx, ny) not in gscore or tentative < gscore[(nx, ny)]:
                came[(nx, ny)] = cur
                gscore[(nx, ny)] = tentative
                f = tentative + heuristic((nx, ny), (gx, gy))
                heapq.heappush(open_heap, (f, (nx, ny)))

    return None


# -------------------------
# Autonomous MAP agent
# -------------------------
@dataclass
class AgentResult:
    occ_px: np.ndarray
    obstacle_is_dark: bool
    threshold: float
    threshold_mode: str
    morph_passes: int
    inflate_px: int
    cell: int
    score: float
    path_cells: list | None


class AutonomousMAPAgent:
    """
    Autonomous, task-aware map segmentation:
      - proposes candidates (threshold/polarity/morph/inflate)
      - validates with connectivity + A* path feasibility
      - scores based on clearance and map quality
    """

    def __init__(
        self,
        cell: int = 4,
        morph_candidates=( 3, 4, 5),
        inflate_candidates=( 4, 8, 12, 16),
        threshold_modes=("otsu", "percentiles"),
        percentiles=(25, 35, 45, 55, 65, 75),
    ):
        self.cell = int(cell)
        self.morph_candidates = tuple(morph_candidates)
        self.inflate_candidates = tuple(inflate_candidates)
        self.threshold_modes = tuple(threshold_modes)
        self.percentiles = tuple(percentiles)

    def solve(self, map_surface: pygame.Surface, start_px, goal_px) -> AgentResult:
        gray = Map2D.surface_to_gray(map_surface)

        # build threshold candidate list
        candidates = []
        if "otsu" in self.threshold_modes:
            candidates.append(("otsu", float(Map2D.otsu_threshold(gray))))
        if "percentiles" in self.threshold_modes:
            for p in self.percentiles:
                candidates.append((f"p{p}", float(np.percentile(gray, p))))

        best = None
        best_score = -1e18

        for mode, thr in candidates:
            for obstacle_is_dark in (True, False):
                base_occ = (gray < thr) if obstacle_is_dark else (gray > thr)

                for mp in self.morph_candidates:
                    occ = base_occ
                    if mp > 0:
                        occ = Map2D.cleanup_static(occ, passes=mp)

                    for inf in self.inflate_candidates:
                        occ2 = occ
                        if inf > 0:
                            occ2 = Map2D.inflate_static(occ2, radius=inf)

                        # ---- task-aware validate on coarse grid ----
                        grid_occ = downsample_occ(occ2.astype(bool), self.cell)
                        s_cell = pix_to_cell(start_px, self.cell)
                        g_cell = pix_to_cell(goal_px, self.cell)

                        free_grid = ~grid_occ
                        if not bfs_connected(free_grid, s_cell, g_cell):
                            continue

                        path = astar_path(grid_occ, s_cell, g_cell)
                        if path is None or len(path) < 2:
                            continue

                        score = self._score_candidate(grid_occ, path)
                        if score > best_score:
                            best_score = score
                            best = AgentResult(
                                occ_px=occ2.astype(bool),
                                obstacle_is_dark=obstacle_is_dark,
                                threshold=float(thr),
                                threshold_mode=str(mode),
                                morph_passes=int(mp),
                                inflate_px=int(inf),
                                cell=int(self.cell),
                                score=float(score),
                                path_cells=path,
                            )

        # If nothing feasible, fall back to “best looking” (non-task-aware) so you still get output
        if best is None:
            best = self._fallback_best_looking(gray)

        return best

    def _score_candidate(self, grid_occ: np.ndarray, path_cells: list) -> float:
        """
        Higher better.
        Rewards:
          - clearance along the path (distance from obstacles)
          - shorter path a bit (don’t wander)
        Penalizes:
          - very noisy maps (rough boundary)
          - too dense obstacles
        """
        h, w = grid_occ.shape
        total = h * w
        obstacle_frac = float(grid_occ.mean())
        if obstacle_frac < 0.02 or obstacle_frac > 0.85:
            return -1e9

        dist = distance_to_obstacles(grid_occ)  # coarse distances
        path_d = np.array([dist[cy, cx] for (cx, cy) in path_cells], dtype=np.float32)

        min_clear = float(np.min(path_d))
        mean_clear = float(np.mean(path_d))
        path_len = float(len(path_cells))

        # roughness: neighbor flips on coarse grid
        flips = (np.abs(np.diff(grid_occ.astype(np.int8), axis=0)).sum() +
                 np.abs(np.diff(grid_occ.astype(np.int8), axis=1)).sum())
        flips_norm = float(flips) / float(total)

        # weights tuned for navigation usefulness
        score = (
            2.5 * min_clear
            + 1.0 * mean_clear
            - 0.02 * path_len
            - 1.2 * flips_norm
            - 1.0 * abs(obstacle_frac - 0.25)
        )
        return float(score)

    def _fallback_best_looking(self, gray: np.ndarray) -> AgentResult:
        """If task-aware search fails, pick a non-degenerate map anyway."""
        best = None
        best_score = -1e18

        candidates = [("otsu", float(Map2D.otsu_threshold(gray)))]
        for p in self.percentiles:
            candidates.append((f"p{p}", float(np.percentile(gray, p))))

        for mode, thr in candidates:
            for obstacle_is_dark in (True, False):
                base_occ = (gray < thr) if obstacle_is_dark else (gray > thr)
                for mp in self.morph_candidates:
                    occ = base_occ
                    if mp > 0:
                        occ = Map2D.cleanup_static(occ, passes=mp)
                    for inf in self.inflate_candidates:
                        occ2 = occ
                        if inf > 0:
                            occ2 = Map2D.inflate_static(occ2, radius=inf)

                        obstacle_frac = float(occ2.mean())
                        if obstacle_frac < 0.02 or obstacle_frac > 0.85:
                            continue

                        # prefer large free component on coarse grid
                        grid_occ = downsample_occ(occ2.astype(bool), self.cell)
                        free = ~grid_occ
                        lcc = self._largest_component_size(free)
                        flips = (np.abs(np.diff(grid_occ.astype(np.int8), axis=0)).sum() +
                                 np.abs(np.diff(grid_occ.astype(np.int8), axis=1)).sum())
                        flips_norm = float(flips) / float(grid_occ.size)

                        s = 5.0 * (lcc / float(grid_occ.size)) - 1.0 * flips_norm - 1.0 * abs(obstacle_frac - 0.25)
                        if s > best_score:
                            best_score = s
                            best = AgentResult(
                                occ_px=occ2.astype(bool),
                                obstacle_is_dark=obstacle_is_dark,
                                threshold=float(thr),
                                threshold_mode=str(mode),
                                morph_passes=int(mp),
                                inflate_px=int(inf),
                                cell=int(self.cell),
                                score=float(s),
                                path_cells=None,
                            )

        assert best is not None
        return best

    @staticmethod
    def _largest_component_size(free_mask: np.ndarray) -> int:
        """Largest 4-connected component size on coarse free grid."""
        h, w = free_mask.shape
        seen = np.zeros_like(free_mask, dtype=bool)
        best = 0
        ys, xs = np.where(free_mask)
        for y0, x0 in zip(ys, xs):
            if seen[y0, x0]:
                continue
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            size = 0
            while q:
                y, x = q.popleft()
                size += 1
                for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and free_mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            best = max(best, size)
        return best


# -------------------------
# UI / drawing helpers
# -------------------------
def draw_text(screen, text, x=8, y=8, color=(20, 20, 20)):
    font = pygame.font.SysFont(None, 22)
    screen.blit(font.render(text, True, color), (x, y))


def draw_circle(screen, pos, color, r=7, width=0):
    pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), int(r), width)


def draw_path(screen, path_cells, cell, color=(0, 120, 255), width=3):
    if not path_cells or len(path_cells) < 2:
        return
    pts = [cell_to_pix_center(c, cell) for c in path_cells]
    pts_i = [(int(x), int(y)) for (x, y) in pts]
    pygame.draw.lines(screen, color, False, pts_i, width)


def save_outputs(out_dir, occ_px: np.ndarray, params: dict):
    os.makedirs(out_dir, exist_ok=True)

    # Save occupancy as image
    occ_surf = occ_to_surface(occ_px)
    occ_path = os.path.join(out_dir, "occupancy.png")
    pygame.image.save(occ_surf, occ_path)

    # Save occupancy as .npy
    npy_path = os.path.join(out_dir, "occupancy.npy")
    np.save(npy_path, occ_px.astype(np.uint8))

    # Save params JSON
    json_path = os.path.join(out_dir, "params.json")
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"[saved] {occ_path}")
    print(f"[saved] {npy_path}")
    print(f"[saved] {json_path}")


# -------------------------
# Main
# -------------------------
def main():
    pygame.init()

    if len(sys.argv) < 2:
        print("Usage: python autonomous_map_agent.py \"/absolute/path/to/map.png\"")
        sys.exit(1)

    map_path = sys.argv[1]
    if not os.path.exists(map_path):
        print(f"Map file not found: {map_path}")
        sys.exit(1)

    # Load raw first
    raw = pygame.image.load(map_path)

    # Create window then convert
    w, h = raw.get_width(), raw.get_height()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Autonomous MAP Agent (click A then B)")

    map_img = raw.convert()

    agent = AutonomousMAPAgent(
        cell=4,  # planning resolution (px per cell); 4 is a good start
        morph_candidates=(0, 1, 2, 3, 4, 5),
        inflate_candidates=(0, 4, 8, 12, 16),
        threshold_modes=("otsu", "percentiles"),
        percentiles=(25, 35, 45, 55, 65, 75),
    )

    start_px = None
    goal_px = None

    result: AgentResult | None = None
    occ_overlay = None
    show_occ = True

    def rerun():
        nonlocal result, occ_overlay
        if start_px is None or goal_px is None:
            return
        result = agent.solve(map_img, start_px, goal_px)
        occ_overlay = overlay_occ_on_map(map_img, result.occ_px, alpha=110)

        print("\n--- AutonomousMAPAgent chose ---")
        print(json.dumps({
            "score": result.score,
            "threshold_mode": result.threshold_mode,
            "threshold": result.threshold,
            "obstacle_is_dark": result.obstacle_is_dark,
            "morph_passes": result.morph_passes,
            "inflate_px": result.inflate_px,
            "cell": result.cell,
            "path_found": (result.path_cells is not None),
        }, indent=2))

    clock = pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    show_occ = not show_occ
                elif e.key == pygame.K_c:
                    start_px, goal_px = None, None
                    result, occ_overlay = None, None
                elif e.key == pygame.K_r:
                    rerun()
                elif e.key == pygame.K_s:
                    if result is not None:
                        out_dir = os.path.join(os.path.dirname(map_path), "map_agent_outputs")
                        params = asdict(result)
                        # occ is large; don't dump full array into JSON
                        params.pop("occ_px", None)
                        save_outputs(out_dir, result.occ_px, params)

            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                if start_px is None:
                    start_px = (mx, my)
                elif goal_px is None:
                    goal_px = (mx, my)
                    rerun()
                else:
                    # if both exist, clicking again resets goal (quick adjustment)
                    goal_px = (mx, my)
                    rerun()

        # draw background
        if show_occ and occ_overlay is not None:
            screen.blit(occ_overlay, (0, 0))
        else:
            screen.blit(map_img, (0, 0))

        # draw start/goal
        if start_px is not None:
            draw_circle(screen, start_px, (0, 180, 0), r=8)
            draw_text(screen, "A", int(start_px[0] + 10), int(start_px[1] - 10), (0, 120, 0))
        if goal_px is not None:
            draw_circle(screen, goal_px, (200, 40, 40), r=8)
            draw_text(screen, "B", int(goal_px[0] + 10), int(goal_px[1] - 10), (160, 0, 0))

        # draw path
        if result is not None and result.path_cells is not None:
            draw_path(screen, result.path_cells, result.cell, color=(0, 120, 255), width=3)

        # HUD
        draw_text(screen, "Click A then B | SPACE: toggle | R: rerun | C: clear | S: save | ESC: quit", 8, 8)
        if result is not None:
            draw_text(
                screen,
                f"mode={result.threshold_mode} thr={result.threshold:.1f} dark={result.obstacle_is_dark} "
                f"morph={result.morph_passes} infl={result.inflate_px} cell={result.cell} score={result.score:.3f}",
                8, 30
            )
            if result.path_cells is None:
                draw_text(screen, "No path found with task-aware search (fallback shown).", 8, 52, (160, 0, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
