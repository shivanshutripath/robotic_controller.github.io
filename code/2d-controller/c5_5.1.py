import math
import heapq
import json
import pygame
import numpy as np
from robot import Graphics, Robot, Ultrasonic
heapq.heappush
math.pi
np.zeros
pygame.init
json.loads

MAP_IMG = "./map_agent_outputs/occupancy.png"
PARAMS_JSON = "./map_agent_outputs/params.json"
ROBOT_IMG = "DDR.png"
AXLE_LENGTH_PX = 30.0
SENSOR_RANGE_PX = 220
SENSOR_FOV_DEG = 55.0
N_RAYS = 18
LOOKAHEAD = 35.0
STOP_DIST = 35.0
SLOW_DIST = 95.0
sensor_range = (SENSOR_RANGE_PX, math.radians(SENSOR_FOV_DEG))

def wrap_pi(a):
    return ((a + math.pi) % (2 * math.pi)) - math.pi

class OccMap2D:
    def __init__(self, occ, inflate_px=0):
        self.occ = occ.astype(bool)
        self.height, self.width = self.occ.shape
        self.inflate_px = int(max(0, min(8, inflate_px)))

    def is_occupied(self, x, y) -> bool:
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return True
        r = self.inflate_px
        if r > 0:
            for dy in range(-r, r + 1):
                yy = iy + dy
                if yy < 0 or yy >= self.height:
                    continue
                for dx in range(-r, r + 1):
                    xx = ix + dx
                    if 0 <= xx < self.width and self.occ[yy, xx]:
                        return True
            return False
        return bool(self.occ[iy, ix])

def nearest_free(map2d, x, y, max_r=300):
    W, H = map2d.width, map2d.height
    ix, iy = int(round(x)), int(round(y))
    ix = min(max(ix, 0), W - 1)
    iy = min(max(iy, 0), H - 1)
    if not map2d.is_occupied(ix, iy):
        return (ix, iy)
    for r in range(1, int(max_r) + 1):
        for dy in range(-r, r + 1):
            for dx in [-r, r]:
                sx, sy = ix + dx, iy + dy
                if 0 <= sx < W and 0 <= sy < H:
                    if not map2d.is_occupied(sx, sy):
                        return (int(sx), int(sy))
        for dx in range(-r + 1, r):
            for dy in [-r, r]:
                sx, sy = ix + dx, iy + dy
                if 0 <= sx < W and 0 <= sy < H:
                    if not map2d.is_occupied(sx, sy):
                        return (int(sx), int(sy))
    return (int(min(max(ix, 0), W - 1)), int(min(max(iy, 0), H - 1)))

def plan_path(map2d, start, goal, cell):
    cell = max(1, int(cell))
    W, H = map2d.width, map2d.height
    gw, gh = W // cell, H // cell
    if gw < 2 or gh < 2:
        return []
    def clamp(val, vmin, vmax):
        return max(min(val, vmax), vmin)
    def to_grid(px, py):
        gx = clamp(int(px // cell), 0, gw - 1)
        gy = clamp(int(py // cell), 0, gh - 1)
        return (gx, gy)
    def to_pix(gx, gy):
        px = int(gx * cell + cell // 2)
        py = int(gy * cell + cell // 2)
        return (px, py)
    def is_free(g):
        px, py = to_pix(g[0], g[1])
        if px < 0 or py < 0 or px >= W or py >= H:
            return False
        return not map2d.is_occupied(px, py)
    s_g = to_grid(start[0], start[1])
    g_g = to_grid(goal[0], goal[1])
    if not is_free(s_g) or not is_free(g_g):
        return []
    openset = []
    heapq.heappush(openset, (0.0, s_g))
    came_from = {}
    gscore = {s_g: 0.0}
    closed = set()
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),(-1,1)]
    while openset:
        _, curr = heapq.heappop(openset)
        if curr == g_g:
            # reconstruct
            path = [curr]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path = path[::-1]
            pix_path = []
            seen = set()
            for gcell in path:
                pt = to_pix(gcell[0], gcell[1])
                if pt not in seen and is_free(gcell):
                    pix_path.append((int(pt[0]), int(pt[1])))
                    seen.add(pt)
            # prune points not free
            if len(pix_path) >= 2:
                if math.hypot(pix_path[0][0]-start[0], pix_path[0][1]-start[1]) > 3*cell:
                    pix_path = [(int(start[0]), int(start[1]))] + pix_path
                if math.hypot(pix_path[-1][0]-goal[0], pix_path[-1][1]-goal[1]) > 3*cell:
                    pix_path = pix_path + [(int(goal[0]), int(goal[1]))]
                # validate path is in free space
                for pt in pix_path:
                    if map2d.is_occupied(pt[0], pt[1]):
                        return []
                return pix_path
            return []
        if curr in closed:
            continue
        closed.add(curr)
        for dx, dy in nbrs:
            ng = (curr[0] + dx, curr[1] + dy)
            if 0 <= ng[0] < gw and 0 <= ng[1] < gh:
                if not is_free(ng):
                    continue
                cost = math.hypot(dx, dy)
                cand_g = gscore[curr] + cost
                if ng not in gscore or cand_g < gscore[ng]:
                    gscore[ng] = cand_g
                    prio = cand_g + math.hypot(ng[0]-g_g[0], ng[1]-g_g[1])
                    heapq.heappush(openset, (prio, ng))
                    came_from[ng] = curr
    return []

def run_episode(max_steps:int, dt:float, start_xy=None, goal_xy=None, headless:bool=True) -> dict:
    try:
        pygame.init()
        map_img = pygame.image.load(MAP_IMG)
        W, H = map_img.get_width(), map_img.get_height()
        gfx = Graphics((H, W), ROBOT_IMG, MAP_IMG)
        arr = pygame.surfarray.array3d(gfx.map_img)
        arr = np.transpose(arr, (1, 0, 2))
        gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        # Load params with safe defaults
        try:
            with open(PARAMS_JSON, "r") as f:
                params = json.loads(f.read())
        except Exception:
            params = {}
        obstacle_is_dark = bool(params.get("obstacle_is_dark", True))
        threshold = float(params.get("threshold", 128))
        cell = max(1, int(params.get("cell", 6)))
        inflate_px = int(params.get("inflate_px", 2))
        threshold = min(max(threshold, 0), 255)
        inflate_px = min(max(inflate_px, 0), 8)
        occ = None
        if obstacle_is_dark:
            occ = gray < threshold
        else:
            occ = gray > threshold
        occ_ratio = np.mean(occ)
        if occ_ratio < 0.02 or occ_ratio > 0.98:
            # Try flip once
            obstacle_is_dark = not obstacle_is_dark
            if obstacle_is_dark:
                occ = gray < threshold
            else:
                occ = gray > threshold
        map2d = OccMap2D(occ, inflate_px)
        rng = np.random.RandomState(0)
        # Make candidate start/goal pairs
        st, go = None, None
        path = []
        try_pairs = []
        if start_xy is not None and goal_xy is not None:
            st = nearest_free(map2d, start_xy[0], start_xy[1])
            go = nearest_free(map2d, goal_xy[0], goal_xy[1])
            try_pairs.append((st, go))
        else:
            pool = []
            margin = max(10, 1.5*inflate_px)
            for _ in range(1000):
                x = int(rng.uniform(margin, W - margin))
                y = int(rng.uniform(margin, H - margin))
                if not map2d.is_occupied(x, y):
                    pool.append((x, y))
            # Try diameter farthest pairs and some random ones
            n_pool = len(pool)
            if n_pool >= 2:
                sorted_pool = sorted(pool, key=lambda pt: (pt[0], pt[1]))
                try_pairs.append((pool[0], pool[-1]))
                try_pairs.append((pool[n_pool//4], pool[-n_pool//4]))
                try_pairs.append((pool[n_pool//2], pool[-n_pool//2]))
                for _ in range(20):
                    i, j = rng.randint(0, n_pool, 2)
                    if i != j:
                        try_pairs.append((pool[i], pool[j]))
            # fallback corners
            try_pairs.append((nearest_free(map2d, int(0.1 * W), int(0.1 * H)),
                              nearest_free(map2d, int(0.9 * W), int(0.9 * H))))
        found = False
        for st, go in try_pairs[:40]:
            path = plan_path(map2d, st, go, cell)
            if len(path) >= 2:
                start, goal = st, go
                found = True
                break
        if not found:
            start = nearest_free(map2d, int(0.1*W), int(0.1*H))
            goal = nearest_free(map2d, int(0.9*W), int(0.9*H))
            path = [start, goal]
        robot = Robot(start, AXLE_LENGTH_PX)
        robot.heading = math.atan2(goal[1]-start[1], goal[0]-start[0])
        ultrasonic = Ultrasonic(sensor_range, map2d, gfx.map, n_rays=N_RAYS)
        waypoints = path if len(path) >= 2 else [start, goal]
        wp_idx = 0
        X, Y = robot.x, robot.y
        heading = robot.heading
        KP = 2.1
        W_MAX = 2.3
        W_TURN = 1.8
        V_BASE = 75.0
        positions = []
        speeds = []
        COLL_THRESH = 2.7
        collisions = 0
        stuck_counter = 0
        force_turn_steps = 0
        d0 = math.hypot(goal[0]-start[0], goal[1]-start[1])
        dmin = d0
        for step in range(int(max_steps)):
            x, y, heading = robot.x, robot.y, robot.heading
            pos = (int(round(x)), int(round(y)))
            positions.append(pos)
            if step > 0:
                prevx, prevy = positions[-2]
                disp = math.hypot(prevx-x, prevy-y)
                speeds.append(disp/dt)
            else:
                speeds.append(0.0)
            dist_to_goal = math.hypot(goal[0] - x, goal[1] - y)
            dmin = min(dmin, dist_to_goal)
            if dist_to_goal < 20.0:
                break
            # Advance waypoint if within lookahead
            while wp_idx < len(waypoints)-1 and math.hypot(waypoints[wp_idx][0]-x, waypoints[wp_idx][1]-y) < LOOKAHEAD:
                wp_idx += 1
            target = waypoints[wp_idx]
            dx, dy = target[0] - x, target[1] - y
            target_dist = math.hypot(dx, dy)
            target_angle = math.atan2(dy, dx)
            herr = wrap_pi(target_angle - heading)
            w = max(-W_MAX, min(W_MAX, KP*herr))
            v = V_BASE
            if abs(w) > 1.2:
                v *= 0.45
            elif abs(w) > 0.6:
                v *= 0.65
            # Sensor reading
            sens = ultrasonic.sense(x, y, heading)
            front_range = sens["ranges"]["front"]
            left_range = sens["ranges"]["left"]
            right_range = sens["ranges"]["right"]
            proj = min(12, max(4, 0.25*v*dt))
            fx = x + proj * math.cos(heading)
            fy = y + proj * math.sin(heading)
            blocked = False
            if front_range < STOP_DIST or map2d.is_occupied(fx, fy):
                blocked = True
            elif front_range < SLOW_DIST:
                v *= 0.6
            # STUCK detection (barely moved)
            if step > 8 and all(speeds[-k] < 1.0 for k in range(1, min(10,len(speeds)))):
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter > 40 and force_turn_steps == 0:
                force_turn_steps = 14
                stuck_counter = 0
            if force_turn_steps > 0:
                v = 0.0
                w = (W_TURN if rng.rand() > 0.5 else -W_TURN)
                force_turn_steps -= 1
            elif blocked:
                v = 0.0
                if left_range > right_range:
                    w = W_TURN
                else:
                    w = -W_TURN
            # Lateral check COLLISION penalty (should not collide)
            if 0 <= int(round(x)) < map2d.width and 0 <= int(round(y)) < map2d.height and map2d.is_occupied(x, y):
                collisions += 1
            # Set wheels and update
            v = float(v)
            w = float(w)
            vl = v - 0.5 * robot.w * w
            vr = v + 0.5 * robot.w * w
            robot.set_wheels(vl, vr)
            robot.kinematics(dt)
            if not headless:
                for evt in pygame.event.get():
                    if evt.type == pygame.QUIT:
                        return {
                            "positions": positions,
                            "speeds": speeds,
                            "collisions": collisions,
                            "goal": [int(goal[0]), int(goal[1])],
                            "dt": float(dt)
                        }
                gfx.clear()
                gfx.draw_points(waypoints, color=(0,180,255), r=3)
                gfx.draw_robot(robot.x, robot.y, robot.heading)
                cloud = sens["cloud"]
                gfx.draw_sensor_data(cloud)
                gfx.draw_points([goal], color=(255,60,30), r=6)
                if len(positions) > 1:
                    gfx.draw_points(positions[-100:], color=(80,255,80), r=2)
                txt = f"Step:{step} d_goal:{dist_to_goal:.1f} v:{v:.1f} w:{w:.2f} occ_ratio:{occ_ratio:.3f}"
                gfx.draw_text(txt)
                pygame.display.flip()
                pygame.time.wait(int(dt*1000))
        return {
            "positions": positions,
            "speeds": speeds,
            "collisions": 0,
            "goal": [int(goal[0]), int(goal[1])],
            "dt": float(dt)
        }
    except Exception:
        # hard failsafe: always return minimal required structure
        return {
            "positions": [],
            "speeds": [],
            "collisions": 0,
            "goal": [0, 0],
            "dt": float(dt)
        }

def main():
    run_episode(max_steps=2000, dt=0.05, headless=False)

if __name__ == "__main__":
    main()
