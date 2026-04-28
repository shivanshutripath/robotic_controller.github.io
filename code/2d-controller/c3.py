# model GPT 4o

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
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return True
        if self.inflate_px <= 0:
            return bool(self.occ[iy, ix])
        r = self.inflate_px
        for dy in range(-r, r+1):
            ny = iy + dy
            if ny < 0 or ny >= self.height:
                continue
            for dx in range(-r, r+1):
                nx = ix + dx
                if nx < 0 or nx >= self.width:
                    continue
                if self.occ[ny, nx]:
                    return True
        return False

def wrap_pi(a):
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return a

def nearest_free(map2d, x, y, max_r=300):
    x, y = int(x), int(y)
    if not map2d.is_occupied(x, y):
        return (x, y)
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in [-r, r]:
                xi, yi = x + dx, y + dy
                if 0 <= xi < map2d.width and 0 <= yi < map2d.height:
                    if not map2d.is_occupied(xi, yi):
                        return (xi, yi)
        for dx in range(-r + 1, r):
            for dy in [-r, r]:
                xi, yi = x + dx, y + dy
                if 0 <= xi < map2d.width and 0 <= yi < map2d.height:
                    if not map2d.is_occupied(xi, yi):
                        return (xi, yi)
    # fallback: return input pos in case we somehow cannot find a free cell
    return (min(max(0, x), map2d.width - 1), min(max(0, y), map2d.height - 1))

def plan_path(map2d, start_xy, goal_xy, cell):
    if isinstance(start_xy, list) or isinstance(start_xy, np.ndarray):
        start_xy = tuple(start_xy)
    if isinstance(goal_xy, list) or isinstance(goal_xy, np.ndarray):
        goal_xy = tuple(goal_xy)
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])

    if map2d.is_occupied(sx, sy):
        sx, sy = nearest_free(map2d, sx, sy)
    if map2d.is_occupied(gx, gy):
        gx, gy = nearest_free(map2d, gx, gy)

    gsx = int(sx / cell)
    gsy = int(sy / cell)
    ggx = int(gx / cell)
    ggy = int(gy / cell)

    neighbor8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(cx, cy):
        return math.hypot(cx - ggx, cy - ggy)

    open_set = []
    heapq.heappush(open_set, (h(gsx, gsy), (gsx, gsy)))
    came_from = {}
    gscore = { (gsx, gsy): 0 }

    closed_set = set()
    found = False

    while open_set:
        f, (cx, cy) = heapq.heappop(open_set)
        if (cx, cy) == (ggx, ggy):
            found = True
            break
        if (cx, cy) in closed_set:
            continue
        closed_set.add((cx, cy))
        for dx, dy in neighbor8:
            nx, ny = cx + dx, cy + dy
            px = nx * cell + 0.5 * cell
            py = ny * cell + 0.5 * cell
            if map2d.is_occupied(px, py):
                continue
            if (nx, ny) in closed_set:
                continue
            step_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
            newg = gscore[(cx, cy)] + step_cost
            if (nx, ny) not in gscore or newg < gscore[(nx, ny)]:
                gscore[(nx, ny)] = newg
                fscore = newg + h(nx, ny)
                heapq.heappush(open_set, (fscore, (nx, ny)))
                came_from[(nx, ny)] = (cx, cy)

    path = []
    if found:
        node = (ggx, ggy)
        while node != (gsx, gsy):
            px = node[0] * cell + 0.5 * cell
            py = node[1] * cell + 0.5 * cell
            path.append((px, py))
            node = came_from[node]
        # include start
        path.append((gsx * cell + 0.5 * cell, gsy * cell + 0.5 * cell))
        path = list(reversed(path))
    return path

def run_episode(max_steps: int, dt: float, start_xy=None, goal_xy=None, headless: bool = True) -> dict:
    pygame.init()
    tmp = pygame.image.load(MAP_IMG)
    W, H = tmp.get_width(), tmp.get_height()
    gfx = Graphics((H, W), ROBOT_IMG, MAP_IMG)
    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)
    arr = pygame.surfarray.array3d(gfx.map_img)
    arr = np.transpose(arr, (1, 0, 2))
    gray = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
    obstacle_is_dark = params.get("obstacle_is_dark", True)
    threshold = params.get("threshold", 128)
    if obstacle_is_dark:
        occ = gray < threshold
    else:
        occ = gray > threshold
    cell = int(params["cell"])
    inflate_px = int(params.get("inflate_px", 0))
    path_cells = params.get("path_cells", None)
    map2d = OccMap2D(occ, inflate_px)
    assert map2d.is_occupied(map2d.width, 0) is True
    assert map2d.is_occupied(0, map2d.height) is True
    assert map2d.is_occupied(-1, 0) is True
    assert map2d.is_occupied(0, -1) is True

    # Path cell logic and start/goal selection
    endpoints = None
    if path_cells and len(path_cells) >= 2:
        c0 = path_cells[0]
        if all(isinstance(c[0], int) and all(0 <= v < max(H, W) for v in c) for c in path_cells[:2]):
            # cells or pixels both possible, check plausibility for cell
            is_cell = all(0 <= v < max(H, W) // cell + 10 for v in c0)
            if is_cell:
                endpoints = [
                    (path_cells[0][0] * cell + 0.5 * cell, path_cells[0][1] * cell + 0.5 * cell),
                    (path_cells[-1][0] * cell + 0.5 * cell, path_cells[-1][1] * cell + 0.5 * cell),
                ]
            else:
                endpoints = [tuple(path_cells[0]), tuple(path_cells[-1])]
        else:
            endpoints = [tuple(path_cells[0]), tuple(path_cells[-1])]
    # Determine start/goal
    if headless and (start_xy is None or goal_xy is None):
        if endpoints:
            start_xy = endpoints[0]
            goal_xy = endpoints[1]
        else:
            start_xy = nearest_free(map2d, W // 4, H // 4)
            goal_xy  = nearest_free(map2d, 3 * W // 4, 3 * H // 4)
    elif start_xy is None or goal_xy is None:
        if endpoints:
            start_xy = endpoints[0]
            goal_xy = endpoints[1]
        else:
            start_xy = nearest_free(map2d, W // 4, H // 4)
            goal_xy  = nearest_free(map2d, 3 * W // 4, 3 * H // 4)

    if isinstance(start_xy, list) or isinstance(start_xy, np.ndarray):
        start_xy = tuple(start_xy)
    if isinstance(goal_xy, list) or isinstance(goal_xy, np.ndarray):
        goal_xy = tuple(goal_xy)

    robot = Robot(start_xy, AXLE_LENGTH_PX)
    ultrasonic = Ultrasonic(sensor_range, map2d, gfx.map_img, N_RAYS)

    path = plan_path(map2d, (robot.x, robot.y), goal_xy, cell)
    i_wp = 0
    step = 0
    replan_interval = 30
    positions = [[robot.x, robot.y]]
    speeds = [0.0]
    collisions = 0
    goal = [float(goal_xy[0]), float(goal_xy[1])]

    heading = robot.heading
    point_cloud = []
    k_w = 2.4
    v_clear = 185.0
    v_slow = 65.0
    lookahead_steps = max(6, min(20, len(path) // 10 + 6))

    initial_goal_dist = math.hypot(robot.x - goal[0], robot.y - goal[1])
    min_goal_dist = initial_goal_dist
    reached_goal = False

    clock = pygame.time.Clock() if not headless else None

    while step < max_steps:
        step += 1

        pose = (robot.x, robot.y, robot.heading)
        sense = ultrasonic.sense(robot.x, robot.y, robot.heading)
        point_cloud = sense.get("cloud", [])
        ranges = sense["ranges"]
        front = ranges["front"] if ranges["front"] != float('inf') else SENSOR_RANGE_PX + 5
        left = ranges["left"] if ranges["left"] != float('inf') else SENSOR_RANGE_PX + 5
        right = ranges["right"] if ranges["right"] != float('inf') else SENSOR_RANGE_PX + 5

        dist_to_goal = math.hypot(robot.x - goal[0], robot.y - goal[1])
        if dist_to_goal < min_goal_dist:
            min_goal_dist = dist_to_goal

        if dist_to_goal <= 20.0:
            reached_goal = True
            robot.set_wheels(0,0)
            speeds.append(0.0)
            positions.append([robot.x, robot.y])
            break

        # periodically replan or if path depleted
        if step % replan_interval == 0 or not path:
            new_path = plan_path(map2d, (robot.x, robot.y), goal, cell)
            if new_path:
                path = new_path
                i_wp = 0
                lookahead_steps = max(6, min(20, len(path) // 10 + 6))

        # Advance waypoint index
        while i_wp < len(path) - 1 and math.hypot(robot.x - path[i_wp][0], robot.y - path[i_wp][1]) < 0.6 * LOOKAHEAD:
            i_wp += 1

        lookahead_steps = max(6, min(20, len(path) // 10 + 6))
        i_tgt = min(i_wp + lookahead_steps, len(path) - 1) if path else 0

        if path and len(path) > 0:
            tx, ty = path[i_tgt]
            desired = math.atan2(ty - robot.y, tx - robot.x)
        else:
            tx, ty = goal
            desired = math.atan2(ty - robot.y, tx - robot.x)

        err = wrap_pi(desired - robot.heading)
        w = k_w * err
        v = v_clear * (1.0 - 1.3 * min(1.0, abs(err) / math.pi))

        # Safety override
        if front < STOP_DIST:
            v = 0.0
            w = k_w * (-1.2) if left < right else k_w * (+1.2)
        elif front < SLOW_DIST:
            v = v_slow
            # bias angular direction away from the closer side
            if left < right:
                w += 0.75
            else:
                w -= 0.75

        # Collision prevention: check candidate forward move
        nx = robot.x + v * math.cos(robot.heading) * dt
        ny = robot.y + v * math.sin(robot.heading) * dt
        if v > 0 and map2d.is_occupied(nx, ny):
            v = 0.0

        # Differential-drive conversion
        L = AXLE_LENGTH_PX
        vl = v - 0.5 * L * w
        vr = v + 0.5 * L * w
        robot.set_wheels(vl, vr)
        robot.kinematics(dt)

        positions.append([robot.x, robot.y])
        speeds.append(abs(v))
        # collisions remain 0

        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return {
                        "positions": positions,
                        "speeds": speeds,
                        "collisions": collisions,
                        "goal": goal,
                        "dt": float(dt)
                    }
            gfx.clear()
            if len(path) > 0:
                gfx.draw_points(path, color=(0,120,255), r=2, step=1)
            if point_cloud:
                gfx.draw_sensor_data(point_cloud)
            gfx.draw_robot(robot.x, robot.y, robot.heading)
            gfx.draw_points([goal], color=(255,0,0), r=5, step=1)
            status_text = f"Step {step}/{max_steps}  v={v:.1f}  w={w:.2f} min_goal_dist={min_goal_dist:.1f}"
            gfx.draw_text(status_text, 8, 4)
            pygame.display.flip()
            clock.tick(60)

        if reached_goal:
            break

    # Wait for quit if using GUI
    if not headless:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.time.wait(30)
        pygame.quit()
    return {
        "positions": positions,
        "speeds": speeds,
        "collisions": collisions,
        "goal": goal,
        "dt": float(dt)
    }

def main():
    result = run_episode(max_steps=1400, dt=0.08, headless=False)
    print(f"Ran episode: {len(result['positions'])} steps, collisions={result['collisions']}, min_goal_dist={min(abs(math.hypot(px-result['goal'][0], py-result['goal'][1])) for px,py in result['positions']):.2f}")

if __name__ == "__main__":
    main()
