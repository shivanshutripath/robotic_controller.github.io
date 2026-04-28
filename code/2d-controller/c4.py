import math
import heapq
import json
import pygame
import numpy as np
from robot import Graphics, Robot, Ultrasonic

# --- fail-fast references ---
heapq.heappush
math.pi
np.zeros
pygame.init
json.loads

# --- constants ---
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

# --- OccMap2D ---
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

# --- wrap_pi ---
def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

# --- nearest_free ---
def nearest_free(map2d, x, y, max_r=300):
    x, y = int(round(x)), int(round(y))
    if 0<=x<map2d.width and 0<=y<map2d.height and not map2d.is_occupied(x, y):
        return (x, y)
    for r in range(1, max_r+1):
        # left/right edges
        for dy in range(-r, r+1):
            for dx in [-r, r]:
                nx, ny = x+dx, y+dy
                if 0<=nx<map2d.width and 0<=ny<map2d.height:
                    if not map2d.is_occupied(nx, ny):
                        return (nx, ny)
        # top/bottom edges
        for dx in range(-r+1, r):
            for dy in [-r, r]:
                nx, ny = x+dx, y+dy
                if 0<=nx<map2d.width and 0<=ny<map2d.height:
                    if not map2d.is_occupied(nx, ny):
                        return (nx, ny)
    nx = min(max(x, 0), map2d.width-1)
    ny = min(max(y, 0), map2d.height-1)
    return (nx, ny)

# --- plan_path ---
def plan_path(map2d, start_xy, goal_xy, cell):
    def as_tuple(xy):
        if isinstance(xy, (np.ndarray, list)): return tuple(xy)
        return xy
    sx, sy = int(round(as_tuple(start_xy)[0])), int(round(as_tuple(start_xy)[1]))
    gx, gy = int(round(as_tuple(goal_xy)[0])), int(round(as_tuple(goal_xy)[1]))
    if map2d.is_occupied(sx, sy):
        sx, sy = nearest_free(map2d, sx, sy)
    if map2d.is_occupied(gx, gy):
        gx, gy = nearest_free(map2d, gx, gy)
    gsx, gsy = int(sx // cell), int(sy // cell)
    ggx, ggy = int(gx // cell), int(gy // cell)
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
    def h(cx,cy): return math.hypot(cx-ggx, cy-ggy)
    open_set = []
    heapq.heappush(open_set, (h(gsx,gsy), (gsx,gsy)))
    gscore = { (gsx,gsy): 0 }
    came_from = {}
    closed_set = set()
    found = False
    while open_set:
        _, (cx,cy) = heapq.heappop(open_set)
        if (cx,cy)==(ggx,ggy):
            found = True
            break
        if (cx,cy) in closed_set:
            continue
        closed_set.add((cx,cy))
        for dx,dy in neighbors:
            nx, ny = cx+dx, cy+dy
            px = nx*cell + 0.5*cell
            py = ny*cell + 0.5*cell
            if map2d.is_occupied(px, py):
                continue
            step = math.hypot(dx, dy)
            ng = gscore[(cx,cy)] + step
            if (nx,ny) in gscore and ng >= gscore[(nx,ny)]:
                continue
            gscore[(nx,ny)] = ng
            heapq.heappush(open_set, (ng + h(nx,ny), (nx,ny)))
            came_from[(nx,ny)] = (cx,cy)
    if not found:
        return []
    node = (ggx, ggy)
    path = []
    while node != (gsx, gsy):
        x, y = node[0]*cell+0.5*cell, node[1]*cell+0.5*cell
        path.append((x, y))
        node = came_from.get(node)
        if node is None:
            return []
    path.append((gsx*cell+0.5*cell, gsy*cell+0.5*cell))
    path.reverse()
    return path

# --- run_episode ---
def run_episode(max_steps:int, dt:float, start_xy=None, goal_xy=None, headless:bool=True) -> dict:
    pygame.init()
    tmp = pygame.image.load(MAP_IMG)
    W, H = tmp.get_width(), tmp.get_height()
    gfx = Graphics((H, W), ROBOT_IMG, MAP_IMG)
    with open(PARAMS_JSON,'r') as f:
        params = json.load(f)
    arr = pygame.surfarray.array3d(gfx.map_img)
    arr = np.transpose(arr, (1,0,2))
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
    endpoints = None
    if path_cells and len(path_cells) >= 2:
        c0 = path_cells[0]
        is_cell = False
        if isinstance(c0, (list,tuple)) and all(isinstance(v,int) for v in c0):
            if all(0 <= v < max(H,W)//cell + 10 for v in c0):
                is_cell = True
        if is_cell:
            endpoints = [
                (path_cells[0][0]*cell+0.5*cell, path_cells[0][1]*cell+0.5*cell),
                (path_cells[-1][0]*cell+0.5*cell, path_cells[-1][1]*cell+0.5*cell),
            ]
        else:
            endpoints = [tuple(path_cells[0]), tuple(path_cells[-1])]
    else:
        endpoints = None
    # determine start/goal and nearest_free
    if (headless and (start_xy is None or goal_xy is None)) or (start_xy is None or goal_xy is None):
        if endpoints:
            start_xy, goal_xy = tuple(endpoints[0]), tuple(endpoints[1])
        else:
            start_xy = (W//4, H//4)
            goal_xy = (3*W//4, 3*H//4)
    # snap both
    start_xy = tuple(nearest_free(map2d, *start_xy))
    goal_xy = tuple(nearest_free(map2d, *goal_xy))
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
    v_slow  = 65.0
    lookahead_steps = max(6, min(20, len(path)//10 + 6))
    initial_goal_dist = math.hypot(robot.x-goal[0], robot.y-goal[1])
    min_goal_dist = initial_goal_dist
    reached_goal = False
    clock = pygame.time.Clock() if not headless else None
    cmd_gain = 1.0
    stuck_steps = 0
    stuck_window = []
    overshoot_count = 0
    last_goal_dist = math.hypot(robot.x-goal[0], robot.y-goal[1])
    overshoot_mode = False
    finish_behavior = False
    finish_count = 0
    goal_track_buffer = []
    # --- main loop ---
    while step < max_steps:
        step += 1
        pose = (robot.x, robot.y, robot.heading)
        sense = ultrasonic.sense(robot.x, robot.y, robot.heading)
        point_cloud = sense.get("cloud", [])
        ranges = sense["ranges"]
        front = ranges["front"] if ranges["front"] != float('inf') else SENSOR_RANGE_PX+5
        left  = ranges["left"]  if ranges["left"]  != float('inf') else SENSOR_RANGE_PX+5
        right = ranges["right"] if ranges["right"] != float('inf') else SENSOR_RANGE_PX+5
        # handle 'meters' normalization if returned
        for sector in ("front", "left", "right"):
            d = ranges.get(sector)
            if d is None:
                if sector=="front": front = SENSOR_RANGE_PX+5
                if sector=="left": left = SENSOR_RANGE_PX+5
                if sector=="right": right = SENSOR_RANGE_PX+5
            elif d <= 3.0:
                val = d * SENSOR_RANGE_PX
                if sector=="front": front=val
                elif sector=="left": left=val
                elif sector=="right": right=val
        dist_to_goal = math.hypot(robot.x-goal[0], robot.y-goal[1])
        if dist_to_goal < min_goal_dist:
            min_goal_dist = dist_to_goal
        # Direct GOAL seek if within 140px or path is empty or close to goal (critical fix)
        # If very close, always direct seek
        GOAL_DOCK_DIST = 140
        # Replan
        if (step % replan_interval == 0) or (not path):
            new_path = plan_path(map2d, (robot.x, robot.y), goal, cell)
            if new_path:
                path = new_path
                i_wp = 0
                lookahead_steps = max(6, min(20, len(path)//10 + 6))
        # Waypoint advance
        while i_wp < len(path)-1 and math.hypot(robot.x-path[i_wp][0], robot.y-path[i_wp][1]) < 0.6*LOOKAHEAD:
            i_wp += 1
        lookahead_steps = max(6, min(20, len(path)//10 + 6))
        i_tgt = min(i_wp + lookahead_steps, len(path)-1) if path else 0
        # Always explicitly drive toward the provided goal position
        tx, ty = goal
        goal_vec_x = tx - robot.x
        goal_vec_y = ty - robot.y
        desired_goal_heading = math.atan2(goal_vec_y, goal_vec_x)
        # If within docking distance, skip waypoints, drive directly to goal
        # Otherwise, only use waypoints if not docking
        using_waypoints = (dist_to_goal > GOAL_DOCK_DIST) and (path and len(path)>0)
        if using_waypoints:
            tx, ty = path[i_tgt]
            waypoint_vec_x = tx - robot.x
            waypoint_vec_y = ty - robot.y
            desired = math.atan2(waypoint_vec_y, waypoint_vec_x)
        else:
            desired = desired_goal_heading
        err = wrap_pi(desired - robot.heading)
        w = k_w * err
        # --- Adaptive cmd_gain ---
        dx = robot.x - positions[-1][0]
        dy = robot.y - positions[-1][1]
        measured_speed = math.hypot(dx, dy)/dt
        speeds.append(measured_speed)
        positions.append([robot.x, robot.y])
        stuck_window.append([robot.x, robot.y])
        if len(stuck_window) > 60:
            stuck_window.pop(0)
        if front > 1.5*SLOW_DIST and measured_speed < 20:
            stuck_steps += 1
            if stuck_steps >= 25:
                cmd_gain = min(cmd_gain*1.35, 250.0)
        else:
            stuck_steps = 0
            cmd_gain = max(1.0, cmd_gain*0.99)
        # Stuck recovery: force v, reduce |w| if moved <5px last 60 steps, but can go forward
        stuck_dist = math.hypot(robot.x - stuck_window[0][0], robot.y - stuck_window[0][1]) if len(stuck_window)>=60 else 1e6
        stuck_recover = stuck_dist < 5.0 and front > 1.2*STOP_DIST
        # Overshoot prevention
        if step > 5:
            if dist_to_goal > last_goal_dist + 12 and dist_to_goal > 140:
                overshoot_count += 1
            else:
                overshoot_count = 0
            if overshoot_count >= 4:
                overshoot_mode = True
            elif overshoot_count == 0:
                overshoot_mode = False
        last_goal_dist = dist_to_goal
        # --- SPEED POLICY adaptive ---
        # Policy: far=fast, close=slow, sharp turns=slow
        base_v_clear = v_clear
        base_v_slow = v_slow
        # (tighter at sharper turns)
        if abs(err) > 0.75:
            base_v_clear = 120.0
            base_v_slow = 45.0
        # Policy: scale v by goal distance
        if dist_to_goal > 150:
            base_v_clear = 185.0
        elif dist_to_goal > 50:
            base_v_clear = 130.0
        elif dist_to_goal > 25:
            base_v_clear = 80.0
        else:
            base_v_clear = 36.0 # softer crawl speed near goal
        if dist_to_goal < 18.0:
            base_v_clear = 0.0
        v = base_v_clear * (1.0 - 1.3*min(1.0, abs(err)/math.pi))
        # --- Safety override ---
        if front < STOP_DIST:
            v = 0.0
            w = -k_w*1.2 if left < right else +k_w*1.2
        elif front < SLOW_DIST:
            v = base_v_slow
            if left < right:
                w += 0.75
            else:
                w -= 0.75
        # --- Stuck recovery ---
        if stuck_recover:
            v = max(160.0, v)
            w = 0.15 * w
        v *= cmd_gain
        # Overshoot reacquire - reduce v, increase turn
        if overshoot_mode:
            v = v * 0.7
            w = w * 1.45
        # --- Collision prevention ---
        nx = robot.x + v*math.cos(robot.heading)*dt
        ny = robot.y + v*math.sin(robot.heading)*dt
        if v > 0 and map2d.is_occupied(nx, ny):
            v = 0.0
        # --- FINISH behavior: brake to stop at goal ---
        if dist_to_goal <= 20.0:
            reached_goal = True
            robot.set_wheels(0,0)
            robot.kinematics(dt)
            dx = robot.x - positions[-1][0]
            dy = robot.y - positions[-1][1]
            speeds[-1] = math.hypot(dx, dy)/dt
            # no positions.append: already done this step
            break
        # --- Differential drive ---
        L = AXLE_LENGTH_PX
        vl = v - 0.5*L*w
        vr = v + 0.5*L*w
        robot.set_wheels(vl, vr)
        robot.kinematics(dt)
        # --- GUI ---
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return {
                        "positions": positions,
                        "speeds": speeds,
                        "collisions": collisions,
                        "goal": goal,
                        "dt": float(dt),
                    }
            gfx.clear()
            if len(path)>0:
                gfx.draw_points(path, color=(0,120,255), r=2, step=1)
            if point_cloud:
                gfx.draw_sensor_data(point_cloud)
            gfx.draw_robot(robot.x, robot.y, robot.heading)
            gfx.draw_points([goal], color=(255,0,0), r=5, step=1)
            status_text = f"Step {step}/{max_steps}  v={v:.1f}  w={w:.2f} min_goal_dist={min_goal_dist:.1f}"
            gfx.draw_text(status_text, 8, 4)
            pygame.display.flip()
            clock.tick(60)
    # after loop, keep window open if not headless
    if not headless:
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            pygame.time.wait(30)
        pygame.quit()
    return {
        "positions": positions,
        "speeds": speeds,
        "collisions": collisions,
        "goal": goal,
        "dt": float(dt),
    }

def main():
    result = run_episode(max_steps=1400, dt=0.08, headless=False)
    goal_x, goal_y = result['goal']
    min_goal_dist = min(abs(math.hypot(px-goal_x, py-goal_y)) for px,py in result['positions'])
    print(f"Ran episode: {len(result['positions'])} steps, collisions={result['collisions']}, min_goal_dist={min_goal_dist:.2f}")

if __name__ == "__main__": main()
