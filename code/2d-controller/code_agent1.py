#!/usr/bin/env python3
"""
code_agent.py

Deterministically generates controller.py from:
  - controller_template.py
  - a map image path
  - optional params.json (obstacle_is_dark, morph_passes, inflate_px, cell)

No pygame runtime checks here. Just writes controller.py.

Usage:
  python code_agent.py --template controller_template.py --map ./map.png --out controller.py
  python code_agent.py --template controller_template.py --map ./map.png --params ./params.json --out controller.py
"""

import argparse
import json
from pathlib import Path

ANCHOR_BEGIN = "# === BEGIN GENERATED CONTROLLER ==="
ANCHOR_END = "# === END GENERATED CONTROLLER ==="


CONTROLLER_BODY = r'''
# controller.py (auto-generated)
import math
import heapq
import argparse
import pygame
import numpy as np

from robot import Graphics, Robot, Ultrasonic, Map2D


# -------------------------
# Helpers
# -------------------------
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def draw_points(surface, pts, color=(0, 120, 255), r=2, step=1):
    if not pts:
        return
    for i, p in enumerate(pts):
        if i % max(1, step) != 0:
            continue
        x, y = int(p[0]), int(p[1])
        pygame.draw.circle(surface, color, (x, y), int(r))


def draw_text(surface, text, x, y, color=(20, 20, 20)):
    font = pygame.font.Font(None, 22)
    img = font.render(str(text), True, color)
    surface.blit(img, (int(x), int(y)))


def nearest_free(map2d: Map2D, x: int, y: int, max_r: int = 300):
    x, y = int(x), int(y)
    if not map2d.is_occupied(x, y):
        return (x, y)

    for r in range(1, max_r + 1):
        # top & bottom edges
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                xx, yy = x + dx, y + dy
                if not map2d.is_occupied(xx, yy):
                    return (xx, yy)

        # left & right edges
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                xx, yy = x + dx, y + dy
                if not map2d.is_occupied(xx, yy):
                    return (xx, yy)

    return (x, y)


# -------------------------
# A* Planner
# -------------------------
def astar(occ, start, goal, step=6):
    H, W = occ.shape

    def snap(p):
        x, y = p
        return int(round(x / step)), int(round(y / step))

    def unsnap(gx, gy):
        return gx * step, gy * step

    def free(gx, gy):
        x, y = unsnap(gx, gy)
        if x < 0 or y < 0 or x >= W or y >= H:
            return False
        return not occ[int(y), int(x)]

    s = snap(start)
    g = snap(goal)
    if not free(*s) or not free(*g):
        return []

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_heap = [(0.0, s)]
    came = {}
    gscore = {s: 0.0}

    while open_heap:
        _, cur = heapq.heappop(open_heap)

        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return [unsnap(px, py) for px, py in path]

        for dx, dy in nbrs:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not free(*nxt):
                continue

            cost = 1.4142 if (dx and dy) else 1.0
            ng = gscore[cur] + cost
            if ng < gscore.get(nxt, 1e18):
                came[nxt] = cur
                gscore[nxt] = ng
                f = ng + h(nxt, g)
                heapq.heappush(open_heap, (f, nxt))

    return []


def plan_via_landmarks(occ, start, landmarks, goal, step=6):
    pts = [start] + list(landmarks) + [goal]
    full = []
    for i in range(len(pts) - 1):
        seg = astar(occ, pts[i], pts[i + 1], step=step)
        if not seg:
            return []
        full.extend(seg if not full else seg[1:])
    return full


# -------------------------
# Follower + Safety
# -------------------------
def compute_follow_vw(pose, path, lookahead=55.0, v_nom=160.0):
    x, y, th = pose
    if not path:
        return 0.0, 0.0

    d = [math.hypot(px - x, py - y) for px, py in path]
    i0 = int(np.argmin(d))

    iL = i0
    acc = 0.0
    while iL + 1 < len(path) and acc < lookahead:
        x1, y1 = path[iL]
        x2, y2 = path[iL + 1]
        acc += math.hypot(x2 - x1, y2 - y1)
        iL += 1

    tx, ty = path[iL]
    desired = math.atan2(ty - y, tx - x)
    err = wrap_pi(desired - th)

    v = v_nom * max(0.25, 1.0 - abs(err) / (math.pi / 2))
    w = 2.4 * err
    return v, w


def safety_override(v, w, ranges, stop_dist=35.0, slow_dist=95.0):
    dL = float(ranges.get("left", 1e9))
    dF = float(ranges.get("front", 1e9))
    dR = float(ranges.get("right", 1e9))

    if dF < stop_dist:
        if dL < dR:
            return 0.0, -2.2
        else:
            return 0.0, +2.2

    if dF < slow_dist:
        v = min(v, 55.0)
        rep = (1.0 / max(dR, 1e-3)) - (1.0 / max(dL, 1e-3))
        rep = max(-0.8, min(0.8, rep))
        w = w - 1.4 * rep
        return v, w

    return v, w


def vw_to_wheels(v, w, wheelbase_px):
    vl = v - 0.5 * wheelbase_px * w
    vr = v + 0.5 * wheelbase_px * w
    return vl, vr


def set_wheels_compat(robot, vl, vr):
    if hasattr(robot, "set_wheels"):
        robot.set_wheels(vl, vr)
    else:
        # fallback: assign if your Robot uses vl/vr fields
        if hasattr(robot, "vl"):
            robot.vl = vl
        if hasattr(robot, "vr"):
            robot.vr = vr


def robot_step_compat(robot, dt):
    if hasattr(robot, "kinematics"):
        robot.kinematics(dt)
    elif hasattr(robot, "step"):
        robot.step()


def sense_compat(sensor, robot):
    # Preferred API: sensor.sense(x,y,heading) -> {'cloud':..., 'ranges':{...}}
    if hasattr(sensor, "sense"):
        return sensor.sense(robot.x, robot.y, robot.heading)

    # Fallback: sensor.sense_obstacles(...) -> cloud list
    if hasattr(sensor, "sense_obstacles"):
        cloud = sensor.sense_obstacles(robot.x, robot.y, robot.heading)
        # crude ranges from cloud
        ranges = {"left": 1e9, "front": 1e9, "right": 1e9}
        for px, py in cloud:
            d = math.hypot(px - robot.x, py - robot.y)
            a = wrap_pi(math.atan2(py - robot.y, px - robot.x) - robot.heading)
            if abs(a) < math.radians(15):
                ranges["front"] = min(ranges["front"], d)
            elif a > 0:
                ranges["left"] = min(ranges["left"], d)
            else:
                ranges["right"] = min(ranges["right"], d)
        return {"cloud": cloud, "ranges": ranges}

    return {"cloud": [], "ranges": {"left": 1e9, "front": 1e9, "right": 1e9}}


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--robot", default="DDR.png")
    args = ap.parse_args()

    # Defaults if params missing
    obstacle_is_dark = True
    morph_passes = 2
    inflate_px = 6
    cell = 5

    if args.params:
        with open(args.params, "r") as f:
            p = json.load(f)
        obstacle_is_dark = bool(p.get("obstacle_is_dark", obstacle_is_dark))
        morph_passes = int(p.get("morph_passes", morph_passes))
        inflate_px = int(p.get("inflate_px", inflate_px))
        cell = int(p.get("cell", cell))

    pygame.init()

    raw_map = pygame.image.load(args.map)  # NO convert here
    H, W = raw_map.get_height(), raw_map.get_width()

    gfx = Graphics((H, W), args.robot, args.map)
    map2d = Map2D(gfx.map_img, obstacle_is_dark=obstacle_is_dark, morph_passes=morph_passes, inflate_px=inflate_px)

    # ---- INITIAL START / GOAL (EDIT THESE) ----
    start_desired = (200, 500)
    goal_desired = (W - 80, H - 80)

    start = nearest_free(map2d, *start_desired)
    goal = nearest_free(map2d, *goal_desired)
    landmarks = []

    # Robot init (wheelbase smaller = tighter turning)
    robot = Robot(start, axle_length_px=30.0)
    robot.heading = 0.0  # 0 right, pi/2 down, pi left, -pi/2 up

    # Ultrasonic (tuple arg!)
    sensor = Ultrasonic(
        sensor_range=(220, math.radians(55)),
        map2d=map2d,
        surface=gfx.map,
        n_rays=18
    )

    astar_step = max(2, int(cell))
    path = plan_via_landmarks(map2d.occ, (robot.x, robot.y), landmarks, goal, step=astar_step)

    clock = pygame.time.Clock()
    last = pygame.time.get_ticks()
    running = True
    show_path = True
    ui_msg = "LMB: goal | RMB: start | MMB: landmark | C: clear | P: toggle path"

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if map2d.is_occupied(mx, my):
                    pass
                else:
                    if event.button == 1:      # goal
                        goal = (mx, my)
                    elif event.button == 3:    # teleport start
                        robot.x, robot.y = float(mx), float(my)
                    elif event.button == 2:    # add landmark
                        landmarks.append((mx, my))
                    path = plan_via_landmarks(map2d.occ, (robot.x, robot.y), landmarks, goal, step=astar_step)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    landmarks = []
                    path = plan_via_landmarks(map2d.occ, (robot.x, robot.y), landmarks, goal, step=astar_step)
                if event.key == pygame.K_p:
                    show_path = not show_path

        now = pygame.time.get_ticks()
        dt = (now - last) / 1000.0
        last = now
        if dt <= 0:
            dt = 1 / 60.0

        gfx.map.blit(gfx.map_img, (0, 0))

        s = sense_compat(sensor, robot)

        pose = (robot.x, robot.y, robot.heading)
        v, w = compute_follow_vw(pose, path, lookahead=55.0, v_nom=160.0)
        v, w = safety_override(v, w, s["ranges"], stop_dist=35.0, slow_dist=95.0)

        vl, vr = vw_to_wheels(v, w, wheelbase_px=getattr(robot, "w", 30.0))
        set_wheels_compat(robot, vl, vr)
        robot_step_compat(robot, dt)

        # Draw overlays
        if show_path and path:
            draw_points(gfx.map, path, color=(0, 120, 255), r=2, step=4)

        if landmarks:
            draw_points(gfx.map, landmarks, color=(255, 200, 0), r=6, step=1)

        draw_points(gfx.map, [goal], color=(0, 255, 0), r=7, step=1)

        gfx.draw_robot(robot.x, robot.y, robot.heading)

        # sensor cloud
        cloud = s.get("cloud", [])
        if hasattr(gfx, "draw_sensor_data"):
            gfx.draw_sensor_data(cloud)
        else:
            draw_points(gfx.map, cloud, color=(255, 0, 0), r=2, step=1)

        mx, my = pygame.mouse.get_pos()
        draw_text(gfx.map, ui_msg, 8, 8)
        draw_text(gfx.map, f"mouse: ({mx},{my})  start:{(int(robot.x),int(robot.y))}  goal:{(int(goal[0]),int(goal[1]))}", 8, 30)

        if not path:
            draw_text(gfx.map, "WARNING: No path found. Try smaller inflate_px or smaller astar_step.", 8, 52, color=(180, 0, 0))

        pygame.display.update()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    import json
    main()
'''.strip("\n")


def inject(template_text: str, body: str) -> str:
    if ANCHOR_BEGIN not in template_text or ANCHOR_END not in template_text:
        raise ValueError(f"Template must contain:\n{ANCHOR_BEGIN}\n{ANCHOR_END}")
    pre, rest = template_text.split(ANCHOR_BEGIN, 1)
    _, post = rest.split(ANCHOR_END, 1)
    return pre + ANCHOR_BEGIN + "\n" + body + "\n" + ANCHOR_END + post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--map", default=None)     # not embedded; controller takes --map at runtime
    ap.add_argument("--params", default=None)  # not embedded; controller takes --params at runtime
    ap.add_argument("--out", default="controller.py")
    args = ap.parse_args()

    template_p = Path(args.template).expanduser().resolve()
    out_p = Path(args.out).expanduser().resolve()

    if not template_p.exists():
        raise FileNotFoundError(template_p)

    template_text = template_p.read_text(encoding="utf-8")
    controller_text = inject(template_text, CONTROLLER_BODY)
    out_p.write_text(controller_text, encoding="utf-8")
    print(f"[OK] Wrote {out_p}")
    print("Run:")
    print("  python controller.py --map <your_map.png> --params <optional_params.json> --robot DDR.png")


if __name__ == "__main__":
    main()
