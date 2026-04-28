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

class OccMap2D:
    def __init__(self, occ, inflate_px=0):
        self.occ = occ.astype(bool)
        self.height, self.width = self.occ.shape
        self.inflate_px = int(max(0, inflate_px))

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

def wrap_pi(a):
    return ((a + math.pi) % (2 * math.pi)) - math.pi

def nearest_free(map2d, x, y, max_r=300):
    x, y = float(x), float(y)
    if not map2d.is_occupied(int(x), int(y)):
        return (int(x), int(y))
    
    for radius in range(1, int(max_r) + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx = int(x + dx)
                ny = int(y + dy)
                if not map2d.is_occupied(nx, ny):
                    return (nx, ny)
    
    cx = int(np.clip(x, 0, map2d.width - 1))
    cy = int(np.clip(y, 0, map2d.height - 1))
    return (cx, cy)

def plan_path(map2d, start, goal, cell):
    cell = max(1, int(cell))
    gw = map2d.width // cell
    gh = map2d.height // cell
    
    if gw < 2 or gh < 2:
        return []
    
    def to_grid(px, py):
        gx = int(px // cell)
        gy = int(py // cell)
        gx = max(0, min(gx, gw - 1))
        gy = max(0, min(gy, gh - 1))
        return (gx, gy)
    
    def to_pix(gx, gy):
        px = int(gx * cell + cell // 2)
        py = int(gy * cell + cell // 2)
        return (px, py)
    
    def is_free(g):
        px, py = to_pix(g[0], g[1])
        return not map2d.is_occupied(px, py)
    
    start_g = to_grid(start[0], start[1])
    goal_g = to_grid(goal[0], goal[1])
    
    if not is_free(start_g) or not is_free(goal_g):
        return []
    
    def heuristic(g):
        return math.hypot(g[0] - goal_g[0], g[1] - goal_g[1])
    
    open_set = []
    heapq.heappush(open_set, (0, start_g))
    came_from = {}
    g_score = {start_g: 0}
    f_score = {start_g: heuristic(start_g)}
    closed = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed:
            continue
        closed.add(current)
        
        if current == goal_g:
            path = []
            node = current
            while node in came_from:
                path.append(to_pix(node[0], node[1]))
                node = came_from[node]
            path.append(to_pix(start_g[0], start_g[1]))
            path.reverse()
            return path
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor[0] < 0 or neighbor[0] >= gw or neighbor[1] < 0 or neighbor[1] >= gh:
                    continue
                if neighbor in closed:
                    continue
                if not is_free(neighbor):
                    continue
                
                tentative_g = g_score[current] + math.hypot(dx, dy)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []

def run_episode(max_steps: int, dt: float, start_xy=None, goal_xy=None, headless: bool = True) -> dict:
    try:
        pygame.init()
        map_img = pygame.image.load(MAP_IMG)
        W, H = map_img.get_width(), map_img.get_height()
        gfx = Graphics((H, W), ROBOT_IMG, MAP_IMG)
        
        arr = pygame.surfarray.array3d(gfx.map_img).astype(np.float32)
        arr = np.transpose(arr, (1, 0, 2))
        gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        
        params = {
            "obstacle_is_dark": True,
            "threshold": 128,
            "cell": 6,
            "inflate_px": 2
        }
        
        try:
            with open(PARAMS_JSON, 'r') as f:
                loaded = json.load(f)
                params["obstacle_is_dark"] = loaded.get("obstacle_is_dark", True)
                params["threshold"] = float(loaded.get("threshold", 128))
                params["cell"] = max(1, int(loaded.get("cell", 6)))
                params["inflate_px"] = max(0, min(8, int(loaded.get("inflate_px", 2))))
        except:
            pass
        
        threshold = float(np.clip(params["threshold"], 0, 255))
        obstacle_is_dark = params["obstacle_is_dark"]
        occ = (gray < threshold) if obstacle_is_dark else (gray > threshold)
        
        occ_ratio = float(np.sum(occ)) / float(occ.size)
        if occ_ratio < 0.02 or occ_ratio > 0.98:
            obstacle_is_dark = not obstacle_is_dark
            occ = (gray < threshold) if obstacle_is_dark else (gray > threshold)
        
        map2d = OccMap2D(occ, inflate_px=params["inflate_px"])
        
        if start_xy is not None:
            start = nearest_free(map2d, start_xy[0], start_xy[1])
        else:
            start = None
        
        if goal_xy is not None:
            goal = nearest_free(map2d, goal_xy[0], goal_xy[1])
        else:
            goal = None
        
        if start is None or goal is None:
            rng = np.random.RandomState(0)
            candidates = [
                (int(0.1 * W), int(0.1 * H)),
                (int(0.9 * W), int(0.1 * H)),
                (int(0.1 * W), int(0.9 * H)),
                (int(0.9 * W), int(0.9 * H)),
                (int(0.5 * W), int(0.5 * H)),
            ]
            
            found_path = False
            for attempt in range(40):
                if start is None:
                    if attempt < len(candidates):
                        start = nearest_free(map2d, candidates[attempt][0], candidates[attempt][1])
                    else:
                        start = nearest_free(map2d, int(rng.randint(0, W)), int(rng.randint(0, H)))
                
                if goal is None:
                    idx = (attempt + len(candidates) // 2) % len(candidates)
                    if idx < len(candidates):
                        goal = nearest_free(map2d, candidates[idx][0], candidates[idx][1])
                    else:
                        goal = nearest_free(map2d, int(rng.randint(0, W)), int(rng.randint(0, H)))
                
                if start and goal and math.hypot(goal[0] - start[0], goal[1] - start[1]) > 50:
                    path = plan_path(map2d, start, goal, params["cell"])
                    if len(path) >= 2:
                        found_path = True
                        break
                
                if attempt % 5 == 0:
                    start = None
                    goal = None
            
            if not found_path:
                if start is None:
                    start = nearest_free(map2d, int(0.1 * W), int(0.1 * H))
                if goal is None:
                    goal = nearest_free(map2d, int(0.9 * W), int(0.9 * H))
        
        path = plan_path(map2d, start, goal, params["cell"])
        
        robot = Robot(start, AXLE_LENGTH_PX)
        robot.heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
        
        ultrasonic = Ultrasonic(sensor_range, map2d, gfx.map, n_rays=N_RAYS)
        
        positions = [start]
        speeds = []
        collisions = 0
        
        waypoints = path if len(path) >= 2 else [start, goal]
        wp_idx = 0
        
        KP = 2.0
        W_MAX = 2.5
        W_TURN = 1.5
        V_BASE = 110.0
        
        stuck_counter = 0
        stuck_turn_active = False
        stuck_turn_duration = 0
        stuck_turn_dir = 0
        
        last_x, last_y = robot.x, robot.y
        last_update_step = 0
        
        for step in range(max_steps):
            sense_data = ultrasonic.sense(robot.x, robot.y, robot.heading)
            ranges = sense_data["ranges"]
            
            d_goal = math.hypot(goal[0] - robot.x, goal[1] - robot.y)
            
            if d_goal < 20:
                robot.stop()
                old_pos = (robot.x, robot.y)
                robot.kinematics(dt)
                new_pos = (robot.x, robot.y)
                speed = math.hypot(new_pos[1] - old_pos[1], new_pos[0] - old_pos[0]) / dt if dt > 0 else 0.0
                speeds.append(float(speed))
                positions.append((int(robot.x), int(robot.y)))
                break
            
            target = None
            if wp_idx < len(waypoints):
                wp = waypoints[wp_idx]
                d_wp = math.hypot(wp[0] - robot.x, wp[1] - robot.y)
                if d_wp < LOOKAHEAD:
                    wp_idx += 1
                if wp_idx < len(waypoints):
                    target = waypoints[wp_idx]
            
            if target is None:
                target = goal
            
            target_angle = math.atan2(target[1] - robot.y, target[0] - robot.x)
            herr = wrap_pi(target_angle - robot.heading)
            w = max(-W_MAX, min(W_MAX, KP * herr))
            
            front_dist = ranges["front"]
            left_dist = ranges["left"]
            right_dist = ranges["right"]
            
            proj = min(12, max(4, 0.25 * V_BASE * dt))
            proj_x = robot.x + proj * math.cos(robot.heading)
            proj_y = robot.y + proj * math.sin(robot.heading)
            proj_occupied = map2d.is_occupied(int(proj_x), int(proj_y))
            
            v = V_BASE
            
            if front_dist < STOP_DIST or proj_occupied:
                v = 0
                if stuck_turn_active:
                    w = stuck_turn_dir * W_TURN
                    stuck_turn_duration -= 1
                    if stuck_turn_duration <= 0:
                        stuck_turn_active = False
                else:
                    if left_dist > right_dist:
                        w = W_TURN
                    else:
                        w = -W_TURN
                    stuck_counter += 1
                    if stuck_counter > 15:
                        stuck_turn_active = True
                        stuck_turn_duration = 10
                        stuck_turn_dir = 1 if left_dist > right_dist else -1
                        stuck_counter = 0
            else:
                stuck_counter = 0
                stuck_turn_active = False
                
                if front_dist < SLOW_DIST:
                    v = V_BASE * (front_dist / SLOW_DIST) * 0.7
                
                if abs(herr) > 0.5:
                    v *= 0.75
            
            robot.set_wheels(v - 0.5 * w * AXLE_LENGTH_PX / 2.0,
                            v + 0.5 * w * AXLE_LENGTH_PX / 2.0)
            
            old_pos = (robot.x, robot.y)
            robot.kinematics(dt)
            new_pos = (robot.x, robot.y)
            
            if map2d.is_occupied(int(robot.x), int(robot.y)):
                robot.x, robot.y = old_pos[0], old_pos[1]
                collisions += 1
                robot.stop()
            
            speed = math.hypot(new_pos[1] - old_pos[1], new_pos[0] - old_pos[0]) / dt if dt > 0 else 0.0
            speeds.append(float(speed))
            positions.append((int(robot.x), int(robot.y)))
            
            displacement = math.hypot(robot.x - last_x, robot.y - last_y)
            if step - last_update_step >= 400:
                if displacement < 0.001:
                    pass
                last_update_step = step
            
            if not headless:
                gfx.clear()
                gfx.draw_points(waypoints, color=(255, 200, 0), r=3)
                gfx.draw_points(positions[-200:], color=(0, 100, 255), r=1)
                gfx.draw_sensor_data(sense_data["cloud"])
                gfx.draw_robot(robot.x, robot.y, robot.heading)
                gfx.draw_text(f"Step: {step} Dist: {d_goal:.1f}", 8, 8)
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if len(positions) > len(speeds):
                            positions = positions[:len(speeds)]
                        return {
                            "positions": positions,
                            "speeds": speeds,
                            "collisions": collisions,
                            "goal": [int(goal[0]), int(goal[1])],
                            "dt": float(dt)
                        }
        
        if len(positions) > len(speeds):
            positions = positions[:len(speeds)]
        
        return {
            "positions": positions,
            "speeds": speeds,
            "collisions": collisions,
            "goal": [int(goal[0]), int(goal[1])],
            "dt": float(dt)
        }
    except Exception as e:
        return {
            "positions": [(0, 0)],
            "speeds": [0.0],
            "collisions": 0,
            "goal": [0, 0],
            "dt": float(dt)
        }

def main():
    run_episode(max_steps=2000, dt=0.05, headless=False)

if __name__ == "__main__":
    main()
