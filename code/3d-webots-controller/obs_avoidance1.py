import math
import os
import re
import heapq
from typing import List, Tuple, Dict, Any


def extract_floats(val) -> List[float]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            try:
                fx = float(x)
                if math.isfinite(fx):
                    out.append(fx)
            except Exception:
                continue
        return out
    if isinstance(val, str):
        try:
            matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", val)
        except Exception:
            return []
        out = []
        for m in matches:
            try:
                fx = float(m)
                if math.isfinite(fx):
                    out.append(fx)
            except Exception:
                continue
        return out
    return []


def _safe_xy_from_translation(trans_str, default_x=0.0, default_y=0.0) -> Tuple[float, float]:
    floats = extract_floats(trans_str)
    x = float(floats[0]) if len(floats) >= 1 else float(default_x)
    y = float(floats[1]) if len(floats) >= 2 else float(default_y)
    if not math.isfinite(x):
        x = float(default_x)
    if not math.isfinite(y):
        y = float(default_y)
    return x, y


def _compute_bounds(start: Dict[str, float], goal: Dict[str, float], obstacles: List[Dict[str, float]], pad: float = 0.6):
    sx = float(start.get("x", 0.0)) if isinstance(start, dict) else 0.0
    sy = float(start.get("y", 0.0)) if isinstance(start, dict) else 0.0
    gx = float(goal.get("x", 0.0)) if isinstance(goal, dict) else 0.0
    gy = float(goal.get("y", 0.0)) if isinstance(goal, dict) else 0.0

    xs = [sx, gx]
    ys = [sy, gy]
    for o in obstacles if isinstance(obstacles, list) else []:
        try:
            ox = float(o.get("x", 0.0))
            oy = float(o.get("y", 0.0))
            osx = float(o.get("sx", 0.1))
            osy = float(o.get("sy", 0.1))
        except Exception:
            continue
        if not (math.isfinite(ox) and math.isfinite(oy) and math.isfinite(osx) and math.isfinite(osy)):
            continue
        halfx = 0.5 * max(0.01, osx)
        halfy = 0.5 * max(0.01, osy)
        xs.extend([ox - halfx, ox + halfx])
        ys.extend([oy - halfy, oy + halfy])

    try:
        x_min = min(xs) - pad
        x_max = max(xs) + pad
        y_min = min(ys) - pad
        y_max = max(ys) + pad
    except Exception:
        x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0

    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
        x_min, x_max = -1.0, 1.0
    if not (math.isfinite(y_min) and math.isfinite(y_max) and y_max > y_min):
        y_min, y_max = -1.0, 1.0

    return [float(x_min), float(x_max), float(y_min), float(y_max)]


def parse_world(world_text: str, params_text: str = "") -> dict:
    wt = world_text if isinstance(world_text, str) else ""
    pt = params_text if isinstance(params_text, str) else ""

    start = {"x": 0.0, "y": 0.0}
    goal = {"x": 0.5, "y": 0.0}
    obstacles: List[Dict[str, float]] = []

    # Start from E-puck translation if present
    try:
        m = re.search(r"\bE-puck\b\s*{([\s\S]*?)}", wt)
        if m:
            block = m.group(1)
            tm = re.search(r"\btranslation\b\s+([^\n\r]+)", block)
            if tm:
                x, y = _safe_xy_from_translation(tm.group(1), 0.0, 0.0)
                start["x"], start["y"] = x, y
    except Exception:
        pass

    # Goal from DEF GOAL Solid
    try:
        gm = re.search(r"\bDEF\s+GOAL\s+Solid\b\s*{([\s\S]*?)}", wt)
        if gm:
            block = gm.group(1)
            tm = re.search(r"\btranslation\b\s+([^\n\r]+)", block)
            if tm:
                x, y = _safe_xy_from_translation(tm.group(1), goal.get("x", 0.5), goal.get("y", 0.0))
                goal["x"], goal["y"] = x, y
    except Exception:
        pass

    # Optional params overrides: "start: x y" or "start=(x,y)" etc.
    try:
        sm = re.search(r"\bstart\b\s*[:=]\s*([^\n\r;]+)", pt) if isinstance(pt, str) else None
        if sm:
            floats = extract_floats(sm.group(1))
            if len(floats) >= 2:
                start["x"], start["y"] = float(floats[0]), float(floats[1])
        gm2 = re.search(r"\bgoal\b\s*[:=]\s*([^\n\r;]+)", pt) if isinstance(pt, str) else None
        if gm2:
            floats = extract_floats(gm2.group(1))
            if len(floats) >= 2:
                goal["x"], goal["y"] = float(floats[0]), float(floats[1])
    except Exception:
        pass

    # WoodenBox obstacles
    try:
        for m in re.finditer(r"\bWoodenBox\b\s*{([\s\S]*?)}", wt):
            block = m.group(1)
            tm = re.search(r"\btranslation\b\s+([^\n\r]+)", block)
            sm = re.search(r"\bsize\b\s+([^\n\r]+)", block)
            x, y = _safe_xy_from_translation(tm.group(1), 0.0, 0.0) if tm else (0.0, 0.0)

            size_or_scale = sm.group(1) if sm else None
            svals = extract_floats(size_or_scale)
            sx = float(svals[0]) if len(svals) >= 1 else 0.1
            sy = float(svals[1]) if len(svals) >= 2 else (sx if len(svals) >= 1 else 0.1)

            if not math.isfinite(sx) or sx <= 0:
                sx = 0.1
            if not math.isfinite(sy) or sy <= 0:
                sy = 0.1

            obstacles.append({"x": float(x), "y": float(y), "sx": float(sx), "sy": float(sy)})
    except Exception:
        pass

    # Rock obstacles
    try:
        for m in re.finditer(r"\bRock\b\s*{([\s\S]*?)}", wt):
            block = m.group(1)
            tm = re.search(r"\btranslation\b\s+([^\n\r]+)", block)
            scm = re.search(r"\bscale\b\s+([^\n\r]+)", block)
            x, y = _safe_xy_from_translation(tm.group(1), 0.0, 0.0) if tm else (0.0, 0.0)

            scale_or_radius = scm.group(1) if scm else None
            sc_vals = extract_floats(scale_or_radius)

            if len(sc_vals) >= 3:
                scx = float(sc_vals[0])
                scy = float(sc_vals[1])
            elif len(sc_vals) == 2:
                scx = float(sc_vals[0])
                scy = float(sc_vals[1])
            elif len(sc_vals) == 1:
                scx = float(sc_vals[0])
                scy = float(sc_vals[0])
            else:
                scx = 1.0
                scy = 1.0

            if not math.isfinite(scx) or scx <= 0:
                scx = 1.0
            if not math.isfinite(scy) or scy <= 0:
                scy = 1.0

            # Approximate rock footprint (diameter) with a base size scaled.
            base_diameter = 0.18
            sx = base_diameter * scx
            sy = base_diameter * scy
            if not math.isfinite(sx) or sx <= 0:
                sx = 0.1
            if not math.isfinite(sy) or sy <= 0:
                sy = 0.1

            obstacles.append({"x": float(x), "y": float(y), "sx": float(max(0.1, sx)), "sy": float(max(0.1, sy))})
    except Exception:
        pass

    # Ensure start/goal valid
    try:
        start["x"] = float(start.get("x", 0.0))
        start["y"] = float(start.get("y", 0.0))
        if not math.isfinite(start["x"]):
            start["x"] = 0.0
        if not math.isfinite(start["y"]):
            start["y"] = 0.0
    except Exception:
        start = {"x": 0.0, "y": 0.0}

    try:
        if goal is None or not isinstance(goal, dict):
            goal = {"x": 0.5, "y": 0.0}
        gx = goal.get("x", 0.5)
        gy = goal.get("y", 0.0)
        goal["x"] = float(gx) if gx is not None else 0.5
        goal["y"] = float(gy) if gy is not None else 0.0
        if not math.isfinite(goal["x"]):
            goal["x"] = 0.5
        if not math.isfinite(goal["y"]):
            goal["y"] = 0.0
    except Exception:
        goal = {"x": 0.5, "y": 0.0}

    bounds = _compute_bounds(start, goal, obstacles, pad=0.6)

    return {
        "plane": "xy",
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "bounds": bounds,
    }


def build_grid(bounds, obstacles, resolution=0.1, inflation=0.0):
    try:
        x_min, x_max, y_min, y_max = bounds
        x_min = float(x_min)
        x_max = float(x_max)
        y_min = float(y_min)
        y_max = float(y_max)
    except Exception:
        x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0

    try:
        resolution = float(resolution)
        if not math.isfinite(resolution) or resolution <= 0:
            resolution = 0.1
    except Exception:
        resolution = 0.1

    try:
        inflation = float(inflation)
        if not math.isfinite(inflation) or inflation < 0:
            inflation = 0.0
    except Exception:
        inflation = 0.0

    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
        x_min, x_max = -1.0, 1.0
    if not (math.isfinite(y_min) and math.isfinite(y_max) and y_max > y_min):
        y_min, y_max = -1.0, 1.0

    w = int((x_max - x_min) / resolution) + 1
    h = int((y_max - y_min) / resolution) + 1
    w = max(2, min(w, 2000))
    h = max(2, min(h, 2000))

    # Tests expect a 2D list addressable as grid[x][y], with len(grid)=w and len(grid[0])=h
    grid = [[0 for _ in range(h)] for _ in range(w)]

    def world_to_cell(x, y):
        cx = int(round((x - x_min) / resolution))
        cy = int(round((y - y_min) / resolution))
        if cx < 0:
            cx = 0
        elif cx >= w:
            cx = w - 1
        if cy < 0:
            cy = 0
        elif cy >= h:
            cy = h - 1
        return cx, cy

    for o in obstacles if isinstance(obstacles, list) else []:
        try:
            ox = float(o.get("x", 0.0))
            oy = float(o.get("y", 0.0))
            sx = float(o.get("sx", 0.1))
            sy = float(o.get("sy", 0.1))
        except Exception:
            continue
        if not (math.isfinite(ox) and math.isfinite(oy) and math.isfinite(sx) and math.isfinite(sy)):
            continue
        sx = max(0.01, sx) + 2.0 * inflation
        sy = max(0.01, sy) + 2.0 * inflation

        halfx, halfy = 0.5 * sx, 0.5 * sy
        x0, x1 = ox - halfx, ox + halfx
        y0, y1 = oy - halfy, oy + halfy
        c0x, c0y = world_to_cell(x0, y0)
        c1x, c1y = world_to_cell(x1, y1)
        xmin_c, xmax_c = (c0x, c1x) if c0x <= c1x else (c1x, c0x)
        ymin_c, ymax_c = (c0y, c1y) if c0y <= c1y else (c1y, c0y)

        for cx in range(xmin_c, xmax_c + 1):
            col = grid[cx]
            for cy in range(ymin_c, ymax_c + 1):
                col[cy] = 1

    return grid


def astar(grid, start_cell, goal_cell) -> list:
    if not isinstance(grid, list) or len(grid) == 0 or not isinstance(grid[0], list) or len(grid[0]) == 0:
        return []
    w = len(grid)
    h = len(grid[0])

    try:
        sx, sy = int(start_cell[0]), int(start_cell[1])
        gx, gy = int(goal_cell[0]), int(goal_cell[1])
    except Exception:
        return []

    def inb(x, y):
        return 0 <= x < w and 0 <= y < h

    if not inb(sx, sy) or not inb(gx, gy):
        return []

    def hfun(x, y):
        return abs(x - gx) + abs(y - gy)

    open_heap = []
    heapq.heappush(open_heap, (hfun(sx, sy), 0, (sx, sy)))
    came_from = {}
    gscore = {(sx, sy): 0}
    closed = set()

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _f, g, (x, y) = heapq.heappop(open_heap)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path.append((x, y))
            path.reverse()
            return path

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if not inb(nx, ny):
                continue
            try:
                if grid[nx][ny] != 0:
                    continue
            except Exception:
                continue
            step_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2.0)
            ng = g + step_cost
            if ng < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = ng
                came_from[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (ng + hfun(nx, ny), ng, (nx, ny)))

    return []


def compute_wheel_speeds(pose_xy, yaw, waypoint_xy, prox) -> Tuple[float, float]:
    try:
        x, y = float(pose_xy[0]), float(pose_xy[1])
    except Exception:
        x, y = 0.0, 0.0
    try:
        tx, ty = float(waypoint_xy[0]), float(waypoint_xy[1])
    except Exception:
        tx, ty = x, y
    try:
        yaw = float(yaw)
    except Exception:
        yaw = 0.0

    dx = tx - x
    dy = ty - y
    target_angle = math.atan2(dy, dx)
    err = (target_angle - yaw + math.pi) % (2.0 * math.pi) - math.pi
    dist = math.hypot(dx, dy)

    base = 4.0
    turn = 6.0 * max(-1.0, min(1.0, err))

    # Simple obstacle avoidance using proximity sensors (expects list of 8 e-puck prox values)
    avoid = 0.0
    steer = 0.0
    if isinstance(prox, (list, tuple)) and len(prox) > 0:
        p = []
        for v in prox:
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            p.append(fv)
        while len(p) < 8:
            p.append(0.0)

        front = max(p[0], p[7], p[1], p[6])
        left = max(p[5], p[6])
        right = max(p[1], p[2])

        # If something close ahead, slow and turn away
        if front > 80.0:
            avoid = min(1.0, (front - 80.0) / 400.0)
            steer = (right - left) / max(1.0, (right + left))
        elif front > 40.0:
            avoid = 0.5 * min(1.0, (front - 40.0) / 400.0)
            steer = (right - left) / max(1.0, (right + left))

    speed = base
    if dist < 0.08:
        speed = 0.0
        turn = 0.0

    if avoid > 0.0:
        speed = max(0.8, base * (1.0 - 0.8 * avoid))
        turn = turn + 5.0 * (-steer) + (6.0 if steer == 0.0 else 0.0)

    left = speed - turn
    right = speed + turn

    # Clamp to e-puck reasonable speed range
    max_speed = 6.28
    left = max(-max_speed, min(max_speed, left))
    right = max(-max_speed, min(max_speed, right))
    return float(left), float(right)


def run_episode(max_steps: int = 3000, resolution: float = 0.1, inflation: float = 0.05, waypoint_stride: int = 3) -> dict:
    try:
        from controller import Supervisor  # type: ignore
    except Exception:
        return {"success": False, "steps": 0, "error": "Webots controller API not available"}

    sup = Supervisor()
    timestep = int(sup.getBasicTimeStep()) if sup.getBasicTimeStep() else 32

    # Read world file text (best effort)
    world_text = ""
    try:
        wpath = sup.getWorldPath()
        if isinstance(wpath, str) and wpath and os.path.isfile(wpath):
            with open(wpath, "r", encoding="utf-8", errors="ignore") as f:
                world_text = f.read()
    except Exception:
        world_text = ""

    parsed = parse_world(world_text, "")
    bounds = parsed.get("bounds", [-1.0, 1.0, -1.0, 1.0])
    obstacles = parsed.get("obstacles", [])

    grid = build_grid(bounds, obstacles, resolution=resolution, inflation=inflation)

    def world_to_grid(x, y):
        x_min, x_max, y_min, y_max = bounds
        try:
            cx = int(round((float(x) - float(x_min)) / float(resolution)))
            cy = int(round((float(y) - float(y_min)) / float(resolution)))
        except Exception:
            cx, cy = 0, 0
        cx = max(0, min(len(grid) - 1, cx))
        cy = max(0, min(len(grid[0]) - 1, cy))
        return (cx, cy)

    def grid_to_world(cx, cy):
        x_min, x_max, y_min, y_max = bounds
        return (float(x_min) + float(cx) * float(resolution), float(y_min) + float(cy) * float(resolution))

    # Get robot node
    robot = None
    try:
        robot = sup.getSelf()
    except Exception:
        robot = None
    if robot is None:
        try:
            robot = sup.getFromDef("EPUCK")  # unlikely
        except Exception:
            robot = None

    # Motors and sensors
    left_motor = sup.getDevice("left wheel motor") if sup.getDevice("left wheel motor") else None
    right_motor = sup.getDevice("right wheel motor") if sup.getDevice("right wheel motor") else None
    if left_motor is not None:
        try:
            left_motor.setPosition(float("inf"))
            left_motor.setVelocity(0.0)
        except Exception:
            pass
    if right_motor is not None:
        try:
            right_motor.setPosition(float("inf"))
            right_motor.setVelocity(0.0)
        except Exception:
            pass

    prox_sensors = []
    for i in range(8):
        try:
            ps = sup.getDevice(f"ps{i}")
        except Exception:
            ps = None
        if ps is not None:
            try:
                ps.enable(timestep)
            except Exception:
                pass
        prox_sensors.append(ps)

    # Determine start/goal from simulation if possible
    start_xy = parsed.get("start", {"x": 0.0, "y": 0.0})
    goal_xy = parsed.get("goal", {"x": 0.5, "y": 0.0})

    # If GOAL exists in scene, use its translation
    try:
        goal_node = sup.getFromDef("GOAL")
        if goal_node is not None:
            tr = goal_node.getField("translation").getSFVec3f()
            if isinstance(tr, (list, tuple)) and len(tr) >= 2:
                goal_xy = {"x": float(tr[0]), "y": float(tr[1])}
    except Exception:
        pass

    # Plan path
    s_cell = world_to_grid(start_xy.get("x", 0.0), start_xy.get("y", 0.0))
    g_cell = world_to_grid(goal_xy.get("x", 0.0), goal_xy.get("y", 0.0))

    # Clear a small neighborhood around start/goal
    try:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x = max(0, min(len(grid) - 1, s_cell[0] + dx))
                y = max(0, min(len(grid[0]) - 1, s_cell[1] + dy))
                grid[x][y] = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x = max(0, min(len(grid) - 1, g_cell[0] + dx))
                y = max(0, min(len(grid[0]) - 1, g_cell[1] + dy))
                grid[x][y] = 0
    except Exception:
        pass

    path_cells = astar(grid, s_cell, g_cell)
    if not path_cells:
        # fallback: go straight to goal
        path_cells = [s_cell, g_cell]

    # Downsample
    stride = int(waypoint_stride) if isinstance(waypoint_stride, int) and waypoint_stride > 0 else 3
    waypoints = []
    for idx in range(0, len(path_cells), stride):
        cx, cy = path_cells[idx]
        waypoints.append(grid_to_world(cx, cy))
    if not waypoints or waypoints[-1] != grid_to_world(g_cell[0], g_cell[1]):
        waypoints.append(grid_to_world(g_cell[0], g_cell[1]))

    wp_i = 0
    success = False
    steps = 0

    def get_pose_yaw():
        if robot is None:
            return (0.0, 0.0), 0.0
        try:
            p = robot.getPosition()
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                px, py = 0.0, 0.0
            else:
                px, py = float(p[0]), float(p[1])
        except Exception:
            px, py = 0.0, 0.0
        # Webots rotation is axis-angle; use orientation matrix instead if available
        yaw = 0.0
        try:
            mat = robot.getOrientation()  # 3x3 row-major list length 9
            if isinstance(mat, (list, tuple)) and len(mat) >= 9:
                # For Z-up world, yaw about Z can be extracted:
                # yaw = atan2(m10, m00) where m10 is row 1 col 0 (index 3), m00 index 0
                yaw = math.atan2(float(mat[3]), float(mat[0]))
        except Exception:
            try:
                r = robot.getRotation()
                if isinstance(r, (list, tuple)) and len(r) == 4:
                    ax, ay, az, angle = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                    # if axis close to z, use sign*angle
                    if abs(az) > 0.8:
                        yaw = angle if az >= 0 else -angle
            except Exception:
                yaw = 0.0
        if not math.isfinite(yaw):
            yaw = 0.0
        return (px, py), yaw

    while steps < int(max_steps):
        if sup.step(timestep) == -1:
            break
        steps += 1

        pose, yaw = get_pose_yaw()
        gx, gy = float(goal_xy.get("x", 0.5)), float(goal_xy.get("y", 0.0))
        if math.hypot(pose[0] - gx, pose[1] - gy) < 0.12:
            success = True
            break

        # Advance waypoint if close
        if wp_i < len(waypoints):
            wx, wy = waypoints[wp_i]
            if math.hypot(pose[0] - wx, pose[1] - wy) < 0.12 and wp_i < len(waypoints) - 1:
                wp_i += 1
        if wp_i >= len(waypoints):
            wp_i = len(waypoints) - 1

        prox_vals = []
        for ps in prox_sensors:
            if ps is None:
                prox_vals.append(0.0)
            else:
                try:
                    prox_vals.append(float(ps.getValue()))
                except Exception:
                    prox_vals.append(0.0)

        waypoint = waypoints[wp_i] if waypoints else (gx, gy)
        l, r = compute_wheel_speeds(pose, yaw, waypoint, prox_vals)

        if left_motor is not None:
            try:
                left_motor.setVelocity(l)
            except Exception:
                pass
        if right_motor is not None:
            try:
                right_motor.setVelocity(r)
            except Exception:
                pass

    # stop
    if left_motor is not None:
        try:
            left_motor.setVelocity(0.0)
        except Exception:
            pass
    if right_motor is not None:
        try:
            right_motor.setVelocity(0.0)
        except Exception:
            pass

    return {"success": bool(success), "steps": int(steps)}


def main() -> None:
    try:
        # Import allowed here; run_episode handles missing Webots as well.
        from controller import Supervisor  # noqa: F401  # type: ignore
    except Exception:
        # When run outside Webots, do nothing.
        return
    run_episode()


if __name__ == "__main__":
    main()
