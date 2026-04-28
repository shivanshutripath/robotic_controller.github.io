# controller_template.py
from string import Template

TEMPLATE = Template(
r'''# controller.py
import math
import heapq
import json
import pygame
import numpy as np

from robot import Graphics, Robot, Ultrasonic


# -------------------------
# Occupancy wrapper (for Ultrasonic)
# -------------------------
class OccMap2D:
    """
    Minimal map wrapper compatible with your Ultrasonic class:
      - occ[y,x] True means obstacle
      - is_occupied(x,y) returns True for obstacles and out-of-bounds
      - width/height attributes
    """
    def __init__(self, occ_bool: np.ndarray):
        self.occ = occ_bool.astype(bool)
        self.height, self.width = self.occ.shape

    def is_occupied(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return bool(self.occ[int(y), int(x)])


def surface_to_gray(surface: pygame.Surface) -> np.ndarray:
    arr = pygame.surfarray.array3d(surface).astype(np.float32)  # (W,H,3)
    arr = np.transpose(arr, (1, 0, 2))                          # (H,W,3)
    gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return gray


def build_occ_from_occ_png(map_surface: pygame.Surface, threshold: float, obstacle_is_dark: bool) -> np.ndarray:
    """
    Convert occupancy.png to boolean obstacle grid using params.json settings.
    """
    gray = surface_to_gray(map_surface)
    occ = (gray < threshold) if obstacle_is_dark else (gray > threshold)
    return occ.astype(bool)


def nearest_free(map2d: OccMap2D, x: int, y: int, max_r: int = 300):
    x, y = int(x), int(y)
    if not map2d.is_occupied(x, y):
        return (x, y)

    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                xx, yy = x + dx, y + dy
                if not map2d.is_occupied(xx, yy):
                    return (xx, yy)

        for dy in range(-r + 1, r):
            for dx in (-r, r):
                xx, yy = x + dx, y + dy
                if not map2d.is_occupied(xx, yy):
                    return (xx, yy)

    return (x, y)


# -------------------------
# A* on a coarse grid (step = cell)
# -------------------------
def astar(occ, start, goal, step=4):
    H, W = occ.shape

    def snap(p):
        x, y = p
        return int(round(x / step)), int(round(y / step))

    def unsnap(gx, gy):
        return gx * step + step * 0.5, gy * step + step * 0.5

    def free(gx, gy):
        x, y = unsnap(gx, gy)
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return False
        return not occ[iy, ix]

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
                heapq.heappush(open_heap, (ng + h(nxt, g), nxt))

    return []


def path_cells_to_pixels(path_cells, cell):
    # path_cells are [cx, cy] grid coords; convert to pixel centers
    pts = []
    for cx, cy in path_cells:
        pts.append((cx * cell + 0.5 * cell, cy * cell + 0.5 * cell))
    return pts


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def compute_follow_vw(pose, path, lookahead=35.0, v_nom=140.0):
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
    dL = ranges["left"]
    dF = ranges["front"]
    dR = ranges["right"]

    dF = dF if math.isfinite(dF) else 1e9
    dL = dL if math.isfinite(dL) else 1e9
    dR = dR if math.isfinite(dR) else 1e9

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


def main():
    MAP_IMG = "$MAP_IMG"
    PARAMS_JSON = "$PARAMS_JSON"
    ROBOT_IMG = "$ROBOT_IMG"

    pygame.init()

    # Load occupancy image (we also display it as the map)
    occ_surf = pygame.image.load(MAP_IMG)
    H, W = occ_surf.get_height(), occ_surf.get_width()

    gfx = Graphics((H, W), ROBOT_IMG, MAP_IMG)

    # Read params.json
    with open(PARAMS_JSON, "r", encoding="utf-8") as f:
        params = json.load(f)

    obstacle_is_dark = bool(params.get("obstacle_is_dark", True))
    threshold = float(params.get("threshold", 128.0))
    cell = int(params.get("cell", 4))
    path_cells = params.get("path_cells", [])

    # Build occupancy grid from occupancy.png + params
    occ_bool = build_occ_from_occ_png(gfx.map_img, threshold=threshold, obstacle_is_dark=obstacle_is_dark)
    map2d = OccMap2D(occ_bool)

    # Initial path from params if available; else we will A*
    path = path_cells_to_pixels(path_cells, cell) if path_cells else []

    # Start/goal from path endpoints if available
    if path:
        start_desired = (int(path[0][0]), int(path[0][1]))
        goal_desired  = (int(path[-1][0]), int(path[-1][1]))
    else:
        start_desired = (80, H - 80)
        goal_desired  = (W - 80, 80)

    start = nearest_free(map2d, *start_desired)
    goal  = nearest_free(map2d, *goal_desired)

    robot = Robot(start, axle_length_px=$AXLE_LENGTH_PX)
    robot.heading = 0.0

    sensor = Ultrasonic(
        sensor_range=($SENSOR_RANGE_PX, math.radians($SENSOR_FOV_DEG)),
        map2d=map2d,
        surface=gfx.map,
        n_rays=$N_RAYS
    )

    # If no path from params, plan now
    if not path:
        path = astar(map2d.occ, (robot.x, robot.y), goal, step=cell)

    clock = pygame.time.Clock()
    last = pygame.time.get_ticks()
    running = True
    show_path = True
    ui = "LMB: set goal | RMB: teleport | P: toggle path"

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if map2d.is_occupied(mx, my):
                    pass
                else:
                    if event.button == 1:  # goal
                        goal = (mx, my)
                        path = astar(map2d.occ, (robot.x, robot.y), goal, step=cell)
                    elif event.button == 3:  # teleport
                        robot.x, robot.y = float(mx), float(my)
                        path = astar(map2d.occ, (robot.x, robot.y), goal, step=cell)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    show_path = not show_path

        now = pygame.time.get_ticks()
        dt = (now - last) / 1000.0
        last = now
        if dt <= 0:
            dt = 1 / 60.0

        gfx.map.blit(gfx.map_img, (0, 0))

        s = sensor.sense(robot.x, robot.y, robot.heading)

        v_nom = min(robot.maxspeed * 0.9, 160.0)
        v, w = compute_follow_vw(robot.pose, path, lookahead=$LOOKAHEAD, v_nom=v_nom)
        v, w = safety_override(v, w, s["ranges"], stop_dist=$STOP_DIST, slow_dist=$SLOW_DIST)

        vl, vr = vw_to_wheels(v, w, wheelbase_px=robot.w)
        robot.set_wheels(vl, vr)
        robot.kinematics(dt)

        if show_path and path:
            gfx.draw_points(path, color=(0, 120, 255), r=2, step=3)

        gfx.draw_points([goal], color=(0, 255, 0), r=6, step=1)
        gfx.draw_robot(robot.x, robot.y, robot.heading)
        gfx.draw_sensor_data(s["cloud"])

        mx, my = pygame.mouse.get_pos()
        gfx.draw_text(ui, 8, 8)
        gfx.draw_text(f"mouse:({mx},{my})  cell:{cell}  thr:{threshold}  dark:{obstacle_is_dark}", 8, 30)

        pygame.display.update()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
'''
)
