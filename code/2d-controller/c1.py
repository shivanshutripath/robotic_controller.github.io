import math
import heapq
import json
import pygame
import numpy as np
from robot import Graphics, Robot, Ultrasonic

# Fail-fast self-check references (required)
heapq.heappush; math.pi; np.zeros; pygame.init; json.loads

MAP_IMG         = "./map_agent_outputs/occupancy.png"
PARAMS_JSON     = "./map_agent_outputs/params.json"
ROBOT_IMG       = "DDR.png"
AXLE_LENGTH_PX  = 30.0
SENSOR_RANGE_PX = 220
SENSOR_FOV_DEG  = 55.0
N_RAYS          = 18
LOOKAHEAD       = 35.0
STOP_DIST       = 35.0
SLOW_DIST       = 95.0

sensor_range = (SENSOR_RANGE_PX, math.radians(SENSOR_FOV_DEG))


class OccMap2D:
    def __init__(self, occ, inflate_px=0):
        self.occ = occ.astype(bool)
        self.height, self.width = self.occ.shape
        self.inflate_px = int(inflate_px)

    def is_occupied(self, x, y) -> bool:
        ix = int(x)
        iy = int(y)
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return True
        r = self.inflate_px
        if r <= 0:
            return bool(self.occ[iy, ix])
        x0 = max(0, ix - r)
        x1 = min(self.width - 1, ix + r)
        y0 = max(0, iy - r)
        y1 = min(self.height - 1, iy + r)
        # any neighbor in square
        sub = self.occ[y0:y1 + 1, x0:x1 + 1]
        return bool(sub.any())


def wrap_pi(a):
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a


def nearest_free(map2d, x, y, max_r=300):
    cx = int(x)
    cy = int(y)
    if not map2d.is_occupied(cx, cy):
        return (int(cx), int(cy))

    max_r = int(max(1, max_r))
    # Expand square rings
    for r in range(1, max_r + 1):
        x0 = cx - r
        x1 = cx + r
        y0 = cy - r
        y1 = cy + r

        # top and bottom edges
        for ix in range(x0, x1 + 1):
            if not map2d.is_occupied(ix, y0):
                return (int(ix), int(y0))
            if not map2d.is_occupied(ix, y1):
                return (int(ix), int(y1))
        # left and right edges (excluding corners already checked)
        for iy in range(y0 + 1, y1):
            if not map2d.is_occupied(x0, iy):
                return (int(x0), int(iy))
            if not map2d.is_occupied(x1, iy):
                return (int(x1), int(iy))

    # If nothing found, clamp to bounds and return something deterministic
    ix = min(max(0, cx), map2d.width - 1)
    iy = min(max(0, cy), map2d.height - 1)
    return (int(ix), int(iy))


def _heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    # Octile distance for 8-connected grid
    return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)


def plan_path(map2d, start_xy, goal_xy, cell) -> list:
    if start_xy is None or goal_xy is None:
        return []

    cell = int(cell) if int(cell) > 0 else 1
    start_xy = (float(start_xy[0]), float(start_xy[1]))
    goal_xy = (float(goal_xy[0]), float(goal_xy[1]))

    sx_px, sy_px = start_xy
    gx_px, gy_px = goal_xy

    # Snap start/goal to nearest free in pixel space if occupied
    if map2d.is_occupied(sx_px, sy_px):
        sx_px, sy_px = nearest_free(map2d, sx_px, sy_px, max_r=300)
    if map2d.is_occupied(gx_px, gy_px):
        gx_px, gy_px = nearest_free(map2d, gx_px, gy_px, max_r=300)

    sx = int(sx_px / cell)
    sy = int(sy_px / cell)
    gx = int(gx_px / cell)
    gy = int(gy_px / cell)

    grid_w = int(math.ceil(map2d.width / cell))
    grid_h = int(math.ceil(map2d.height / cell))

    sx = min(max(0, sx), grid_w - 1)
    gx = min(max(0, gx), grid_w - 1)
    sy = min(max(0, sy), grid_h - 1)
    gy = min(max(0, gy), grid_h - 1)

    start = (sx, sy)
    goal = (gx, gy)

    def grid_free(nx, ny):
        if nx < 0 or ny < 0 or nx >= grid_w or ny >= grid_h:
            return False
        px = nx * cell + 0.5 * cell
        py = ny * cell + 0.5 * cell
        return (not map2d.is_occupied(px, py))

    if not grid_free(start[0], start[1]):
        # try snapping again with more range and recompute
        sx_px, sy_px = nearest_free(map2d, sx_px, sy_px, max_r=600)
        start = (int(sx_px / cell), int(sy_px / cell))
    if not grid_free(goal[0], goal[1]):
        gx_px, gy_px = nearest_free(map2d, gx_px, gy_px, max_r=600)
        goal = (int(gx_px / cell), int(gy_px / cell))

    if start == goal:
        px = start[0] * cell + 0.5 * cell
        py = start[1] * cell + 0.5 * cell
        return [(float(px), float(py))]

    # A* (heap stores (f, (x,y)) only)
    open_heap = []
    gscore = {start: 0.0}
    came = {}
    closed = set()

    heapq.heappush(open_heap, (_heuristic(start, goal), start))

    neigh = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
             (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]

    # Bound iterations to avoid pathological maps
    max_iters = grid_w * grid_h * 4
    iters = 0

    while open_heap and iters < max_iters:
        iters += 1
        _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            break
        closed.add(cur)

        cx, cy = cur
        for dx, dy, step_cost in neigh:
            nx = cx + dx
            ny = cy + dy
            n = (nx, ny)
            if n in closed:
                continue
            if not grid_free(nx, ny):
                continue
            tentative = gscore[cur] + step_cost
            if tentative < gscore.get(n, float("inf")):
                came[n] = cur
                gscore[n] = tentative
                f = tentative + _heuristic(n, goal)
                heapq.heappush(open_heap, (f, n))

    if goal not in came and goal != start:
        # Attempt a direct LOS-ish fallback by searching nearest reached node to goal
        best = None
        best_f = float("inf")
        for node, g in gscore.items():
            f = g + _heuristic(node, goal)
            if f < best_f:
                best_f = f
                best = node
        if best is None:
            # Provide minimal non-empty
            px = sx * cell + 0.5 * cell
            py = sy * cell + 0.5 * cell
            return [(float(px), float(py))]
        goal_recon = best
    else:
        goal_recon = goal

    # Reconstruct including start and goal pixel centers
    path_grid = [goal_recon]
    while path_grid[-1] != start:
        path_grid.append(came[path_grid[-1]])
    path_grid.reverse()

    path_px = []
    for (cx, cy) in path_grid:
        px = cx * cell + 0.5 * cell
        py = cy * cell + 0.5 * cell
        path_px.append((float(px), float(py)))

    # Ensure non-empty and includes goal center if reachable
    if not path_px:
        px = start[0] * cell + 0.5 * cell
        py = start[1] * cell + 0.5 * cell
        path_px = [(float(px), float(py))]

    return path_px


def _parse_params():
    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)

    obstacle_is_dark = bool(params.get("obstacle_is_dark", True))
    threshold = float(params.get("threshold", 128.0))
    cell = int(params.get("cell", 4))
    path_cells = params.get("path_cells", None)
    inflate_px = int(params.get("inflate_px", 0)) if params.get("inflate_px", 0) is not None else 0

    return {
        "obstacle_is_dark": obstacle_is_dark,
        "threshold": threshold,
        "cell": cell,
        "path_cells": path_cells,
        "inflate_px": inflate_px,
    }


def _pathcells_to_start_goal(path_cells, cell, map_w, map_h):
    if not path_cells or not isinstance(path_cells, list) or len(path_cells) < 1:
        return (None, None, None)

    # Determine whether entries look like cell coords
    # Heuristic: small ints within grid bounds
    grid_w = int(math.ceil(map_w / cell))
    grid_h = int(math.ceil(map_h / cell))

    looks_like_cells = True
    sample_n = min(10, len(path_cells))
    for k in range(sample_n):
        p = path_cells[k]
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            looks_like_cells = False
            break
        x, y = p[0], p[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            looks_like_cells = False
            break
        # if not close to ints or huge
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if abs(float(x) - xi) > 1e-6 or abs(float(y) - yi) > 1e-6:
            looks_like_cells = False
            break
        if xi < 0 or yi < 0 or xi >= grid_w or yi >= grid_h:
            looks_like_cells = False
            break

    pts = []
    if looks_like_cells:
        for p in path_cells:
            cx = int(p[0])
            cy = int(p[1])
            pts.append((cx * cell + 0.5 * cell, cy * cell + 0.5 * cell))
    else:
        for p in path_cells:
            pts.append((float(p[0]), float(p[1])))

    start_xy = pts[0] if pts else None
    goal_xy = pts[-1] if pts else None
    return (start_xy, goal_xy, pts)


def run_episode(max_steps: int, dt: float, start_xy=None, goal_xy=None, headless: bool = True) -> dict:
    pygame.init()

    # Load map once WITHOUT convert/convert_alpha to get (W,H)
    map_surface_for_dims = pygame.image.load(MAP_IMG)
    W, H = map_surface_for_dims.get_width(), map_surface_for_dims.get_height()

    # Create Graphics after that (creates display)
    gfx = Graphics((H, W), ROBOT_IMG, MAP_IMG)

    # Only after gfx created, convert gfx.map_img via surfarray
    params = _parse_params()
    obstacle_is_dark = params["obstacle_is_dark"]
    threshold = params["threshold"]
    cell = params["cell"]
    inflate_px = params["inflate_px"]
    path_cells = params["path_cells"]

    arr = pygame.surfarray.array3d(gfx.map_img)
    arr = np.transpose(arr, (1, 0, 2))
    gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    if obstacle_is_dark:
        occ = gray < threshold
    else:
        occ = gray > threshold

    map2d = OccMap2D(occ, inflate_px=inflate_px)

    assert map2d.is_occupied(map2d.width, 0) is True
    assert map2d.is_occupied(0, map2d.height) is True
    assert map2d.is_occupied(-1, 0) is True
    assert map2d.is_occupied(0, -1) is True

    pcs_start, pcs_goal, pcs_pts = _pathcells_to_start_goal(path_cells, cell, W, H)

    # Start/goal selection rules
    if start_xy is None:
        start_xy = pcs_start
    if goal_xy is None:
        goal_xy = pcs_goal

    if headless and (start_xy is None and goal_xy is None):
        if pcs_start is not None and pcs_goal is not None:
            start_xy, goal_xy = pcs_start, pcs_goal
        else:
            start_xy = nearest_free(map2d, W // 4, H // 4)
            goal_xy = nearest_free(map2d, 3 * W // 4, 3 * H // 4)
    else:
        if start_xy is None:
            start_xy = nearest_free(map2d, W // 4, H // 4)
        if goal_xy is None:
            goal_xy = nearest_free(map2d, 3 * W // 4, 3 * H // 4)

    # Snap final start/goal to free pixels
    if map2d.is_occupied(start_xy[0], start_xy[1]):
        start_xy = nearest_free(map2d, start_xy[0], start_xy[1])
    if map2d.is_occupied(goal_xy[0], goal_xy[1]):
        goal_xy = nearest_free(map2d, goal_xy[0], goal_xy[1])

    robot = Robot(start_xy, AXLE_LENGTH_PX)
    ultrasonic = Ultrasonic(sensor_range, map2d, gfx.map, n_rays=N_RAYS)

    positions = []
    speeds = []
    collisions = 0

    clock = pygame.time.Clock()

    path = []
    i_wp = 0
    point_cloud = []

    def dist_to(pt):
        return math.hypot(pt[0] - robot.x, pt[1] - robot.y)

    def reached_goal():
        return (math.hypot(goal_xy[0] - robot.x, goal_xy[1] - robot.y) <= 20.0)

    # initial plan
    path = plan_path(map2d, (robot.x, robot.y), goal_xy, cell)
    i_wp = 0

    for step in range(int(max_steps)):
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    step = int(max_steps)  # break outer
                    break

        # periodic replanning
        if step % 30 == 0 or not path:
            path = plan_path(map2d, (robot.x, robot.y), goal_xy, cell)
            i_wp = 0

        # sensing
        sense = ultrasonic.sense(robot.x, robot.y, robot.heading)
        point_cloud = sense.get("cloud", [])
        ranges = sense.get("ranges", {})
        front = float(ranges.get("front", float("inf")))
        left = float(ranges.get("left", float("inf")))
        right = float(ranges.get("right", float("inf")))

        # controller
        k_w = 2.4
        v_clear = 190.0
        v = 0.0
        w = 0.0

        if path:
            # advance waypoint
            while i_wp < len(path) - 1 and dist_to(path[i_wp]) < 0.6 * LOOKAHEAD:
                i_wp += 1

            lookahead_steps = max(6, min(20, len(path) // 10 + 6))
            i_tgt = min(i_wp + int(lookahead_steps), len(path) - 1)
            tx, ty = path[i_tgt]

            desired = math.atan2(ty - robot.y, tx - robot.x)
            err = wrap_pi(desired - robot.heading)

            # nominal speeds
            v = v_clear * max(0.0, 1.0 - 0.7 * min(1.0, abs(err) / (math.pi / 2)))
            w = k_w * err
        else:
            # steer directly to goal
            desired = math.atan2(goal_xy[1] - robot.y, goal_xy[0] - robot.x)
            err = wrap_pi(desired - robot.heading)
            w = 2.2 * err
            v = 160.0 * max(0.0, 1.0 - min(1.0, abs(err) / (math.pi / 3)))

        # safety override using ultrasonic
        if front < STOP_DIST:
            v = 0.0
            # rotate away from closer side (toward larger clearance)
            if left >= right:
                w = abs(w) if abs(w) > 0.3 else 0.9
            else:
                w = -abs(w) if abs(w) > 0.3 else -0.9
        elif front < SLOW_DIST:
            v = min(v, 70.0)
            # bias turning away from closer side
            bias = 0.8 * (1.0 - max(0.0, min(1.0, (front - STOP_DIST) / max(1e-6, (SLOW_DIST - STOP_DIST)))))
            if left < right:
                w += bias
            else:
                w -= bias

        # collision prevention: block forward motion if next position occupied
        heading = robot.heading
        nx = robot.x + v * math.cos(heading) * float(dt)
        ny = robot.y + v * math.sin(heading) * float(dt)
        if v > 0.0 and map2d.is_occupied(nx, ny):
            v = 0.0

        # translate (v,w) to wheel speeds
        vl = v - 0.5 * w * robot.w
        vr = v + 0.5 * w * robot.w
        robot.set_wheels(vl, vr)
        robot.kinematics(float(dt))

        positions.append([float(robot.x), float(robot.y)])
        speeds.append(float(abs(v)))

        if headless and reached_goal():
            break

        if not headless:
            gfx.clear()
            if pcs_pts:
                gfx.draw_points(pcs_pts, color=(130, 130, 130), r=1, step=2)
            if path:
                gfx.draw_points(path, color=(0, 120, 255), r=2, step=2)
                if 0 <= i_wp < len(path):
                    gfx.draw_points([path[i_wp]], color=(255, 165, 0), r=4, step=1)
            if point_cloud:
                gfx.draw_sensor_data(point_cloud)
            gfx.draw_robot(robot.x, robot.y, robot.heading)
            gfx.draw_points([(goal_xy[0], goal_xy[1])], color=(0, 200, 0), r=5, step=1)
            gfx.draw_text(f"step {step}/{max_steps}  v={v:.1f}  front={front:.1f}  goal_d={math.hypot(goal_xy[0]-robot.x, goal_xy[1]-robot.y):.1f}", x=8, y=8)
            pygame.display.flip()
            clock.tick(60)

    # stop robot
    robot.stop()

    if not headless:
        # keep window open until quit
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            gfx.clear()
            if pcs_pts:
                gfx.draw_points(pcs_pts, color=(130, 130, 130), r=1, step=2)
            if path:
                gfx.draw_points(path, color=(0, 120, 255), r=2, step=2)
            if point_cloud:
                gfx.draw_sensor_data(point_cloud)
            gfx.draw_robot(robot.x, robot.y, robot.heading)
            gfx.draw_points([(goal_xy[0], goal_xy[1])], color=(0, 200, 0), r=5, step=1)
            gfx.draw_text("Close window to exit.", x=8, y=8)
            pygame.display.flip()
            clock.tick(60)

    return {
        "positions": positions,
        "speeds": speeds,
        "collisions": int(collisions),
        "goal": [float(goal_xy[0]), float(goal_xy[1])],
        "dt": float(dt),
    }


def main():
    run_episode(max_steps=2500, dt=1.0 / 60.0, start_xy=None, goal_xy=None, headless=False)


if __name__ == "__main__":
    main()
