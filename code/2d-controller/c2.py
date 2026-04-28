# model 4.1
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

    def is_occupied(self, x, y):
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return True
        if self.inflate_px <= 0:
            return bool(self.occ[iy, ix])
        r = self.inflate_px
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx = ix + dx
                ny = iy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.occ[ny, nx]:
                        return True
        return False

def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def nearest_free(map2d, x, y, max_r=300):
    ix, iy = int(x), int(y)
    if not map2d.is_occupied(ix, iy):
        return (ix, iy)
    # Spiral outwards: Manhattan circles
    for r in range(1, max_r+1):
        for dx in range(-r, r+1):
            for sdy in [-1, 1]:
                dy = sdy*(r - abs(dx))
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < map2d.width and 0 <= ny < map2d.height:
                    if not map2d.is_occupied(nx, ny):
                        return (nx, ny)
        for dy in range(-r+1, r):  # avoid corners double-count
            for sdx in [-1, 1]:
                dx = sdx*(r - abs(dy))
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < map2d.width and 0 <= ny < map2d.height:
                    if not map2d.is_occupied(nx, ny):
                        return (nx, ny)
    # fallback: original pos
    return (ix, iy)

def plan_path(map2d, start_xy, goal_xy, cell):
    # Accept both list/tuple for start_xy/goal_xy
    start_xy = tuple(start_xy)
    goal_xy = tuple(goal_xy)
    sx, sy = start_xy
    gx, gy = goal_xy
    if map2d.is_occupied(sx, sy):
        sx, sy = nearest_free(map2d, sx, sy)
    if map2d.is_occupied(gx, gy):
        gx, gy = nearest_free(map2d, gx, gy)
    cs = int(cell)
    # Convert PIXEL to GRID coords
    start_grid = (int(sx / cs), int(sy / cs))
    goal_grid = (int(gx / cs), int(gy / cs))
    width = map2d.width
    height = map2d.height

    def node_is_blocked(nx, ny):
        px = nx * cs + 0.5 * cs
        py = ny * cs + 0.5 * cs
        return map2d.is_occupied(px, py)
    # Prepare bounds in grid coords
    max_nx = width // cs
    max_ny = height // cs

    # A*: entry = (f, (x, y))
    heap = []
    heapq.heappush(heap, (0 + math.hypot(goal_grid[0]-start_grid[0], goal_grid[1]-start_grid[1]), start_grid))
    came_from = {}
    gscore = {start_grid: 0}
    closed = set()
    neighbors_8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while heap:
        _, curr = heapq.heappop(heap)
        if curr == goal_grid:
            break
        if curr in closed:
            continue
        closed.add(curr)
        cx, cy = curr
        for dx, dy in neighbors_8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < max_nx and 0 <= ny < max_ny):
                continue
            if node_is_blocked(nx, ny):
                continue
            cost = math.hypot(dx, dy)
            neighbor = (nx, ny)
            tentative = gscore[curr] + cost
            if neighbor in gscore and tentative >= gscore[neighbor]:
                continue
            gscore[neighbor] = tentative
            prio = tentative + math.hypot(goal_grid[0]-nx, goal_grid[1]-ny)
            heapq.heappush(heap, (prio, neighbor))
            came_from[neighbor] = curr
    # Path trace-back
    path_grids = []
    node = goal_grid
    while node in came_from:
        path_grids.append(node)
        node = came_from[node]
    if node == start_grid:
        path_grids.append(start_grid)
    path_grids = path_grids[::-1]
    # Output path as pixel coords (center of cell)
    path = []
    for gx, gy in path_grids:
        px = gx*cs + 0.5*cs
        py = gy*cs + 0.5*cs
        path.append((px, py))
    if len(path) == 0:
        # Fallback path is just to the goal pixel
        path = [(sx, sy), (gx, gy)]
    return path

def run_episode(max_steps:int, dt:float, start_xy=None, goal_xy=None, headless:bool=True):
    pygame.init()

    # Load image & get map dims
    map_surface = pygame.image.load(MAP_IMG)
    W, H = map_surface.get_width(), map_surface.get_height()

    gfx = Graphics((H, W), ROBOT_IMG, MAP_IMG)
    # gfx.map_img available
    arr = pygame.surfarray.array3d(gfx.map_img)
    arr = np.transpose(arr, (1,0,2))
    gray = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
    # params
    with open(PARAMS_JSON,'r') as f:
        params = json.load(f)
    cell = int(params.get('cell', 4))
    obstacle_is_dark = bool(params.get('obstacle_is_dark', True))
    threshold = params.get('threshold', 128.0)
    path_cells = params.get('path_cells', None)
    inflate_px = int(params.get('inflate_px', 0))

    if obstacle_is_dark:
        occ = gray < threshold
    else:
        occ = gray > threshold
    occ = occ.astype(bool)
    map2d = OccMap2D(occ, inflate_px=inflate_px)
    assert map2d.is_occupied(map2d.width, 0) is True
    assert map2d.is_occupied(0, map2d.height) is True
    assert map2d.is_occupied(-1, 0) is True
    assert map2d.is_occupied(0, -1) is True

    # start & goal determination (cell-vs-pixel)
    pc = path_cells if (path_cells and len(path_cells)>=2) else None
    def _is_grid_cell(entry):
        if not (isinstance(entry, (list,tuple)) and len(entry)==2): return False
        a,b = entry
        if isinstance(a,int) and isinstance(b,int) and (0<=a<map2d.width//cell) and (0<=b<map2d.height//cell): return True
        return False

    if headless and start_xy is None and goal_xy is None:
        if pc and _is_grid_cell(pc[0]) and _is_grid_cell(pc[-1]):
            start_xy = (pc[0][0]*cell+0.5*cell, pc[0][1]*cell+0.5*cell)
            goal_xy  = (pc[-1][0]*cell+0.5*cell, pc[-1][1]*cell+0.5*cell)
        elif pc:
            start_xy = tuple(pc[0])
            goal_xy  = tuple(pc[-1])
        else:
            start_xy = nearest_free(map2d, W//4, H//4)
            goal_xy  = nearest_free(map2d, 3*W//4, 3*H//4)
    elif pc and _is_grid_cell(pc[0]) and _is_grid_cell(pc[-1]) and (start_xy is None or goal_xy is None):
        if start_xy is None:
            start_xy = (pc[0][0]*cell+0.5*cell, pc[0][1]*cell+0.5*cell)
        if goal_xy is None:
            goal_xy  = (pc[-1][0]*cell+0.5*cell, pc[-1][1]*cell+0.5*cell)
    elif pc and (start_xy is None or goal_xy is None):
        if start_xy is None:
            start_xy = tuple(pc[0])
        if goal_xy is None:
            goal_xy = tuple(pc[-1])
    else:
        # Use as provided, fallback:
        if start_xy is None:
            start_xy = nearest_free(map2d, W//4, H//4)
        if goal_xy is None:
            goal_xy = nearest_free(map2d, 3*W//4, 3*H//4)
    # Clamp to int for robot init
    start_xy = tuple(int(x) for x in start_xy)
    goal_xy  = tuple(int(x) for x in goal_xy)

    robot = Robot(start_xy, AXLE_LENGTH_PX)
    ultrasonic = Ultrasonic(sensor_range, map2d, gfx.map_img, n_rays=N_RAYS)

    # Path plan
    path = plan_path(map2d, (robot.x, robot.y), goal_xy, cell)
    i_wp = 0
    replan_cnt = 0
    positions = []
    speeds = []
    collisions = 0

    k_w = 2.8
    v_max = 190.0
    v_slow = 70.0
    running = True
    reached_goal = False
    clock = pygame.time.Clock() if not headless else None

    for step in range(int(max_steps)):
        # Sensing
        sense = ultrasonic.sense(robot.x, robot.y, robot.heading)
        ranges = sense["ranges"]
        front = ranges["front"] if math.isfinite(ranges["front"]) else SENSOR_RANGE_PX
        left = ranges["left"] if math.isfinite(ranges["left"]) else SENSOR_RANGE_PX
        right = ranges["right"] if math.isfinite(ranges["right"]) else SENSOR_RANGE_PX

        # Replan every 30 steps or if path empty
        if step % 30 == 0 or path is None or len(path) < 2:
            path = plan_path(map2d, (robot.x, robot.y), goal_xy, cell)
            i_wp = 0
        # Advance waypoint
        def dist(a, b):
            dx, dy = a[0]-b[0], a[1]-b[1]
            return math.hypot(dx, dy)
        while i_wp < len(path)-1 and dist((robot.x, robot.y), path[i_wp]) < 0.6*LOOKAHEAD:
            i_wp += 1
        lookahead_steps = max(6, min(20, len(path)//10 + 6))
        i_tgt = min(i_wp + lookahead_steps, len(path)-1)
        tx, ty = path[i_tgt]
        # Control logic
        x, y, heading = robot.x, robot.y, robot.heading
        if reached_goal:
            robot.set_wheels(0, 0)
        else:
            to_goal = math.hypot(robot.x - goal_xy[0], robot.y - goal_xy[1]) <= 20.0
            if to_goal:
                reached_goal = True
                robot.set_wheels(0,0)
            elif len(path) < 2:
                # Path failed: steer to goal directly
                dx, dy = goal_xy[0] - x, goal_xy[1] - y
                desired = math.atan2(dy, dx)
                err = wrap_pi(desired - heading)
                v = v_max * (0.60 if abs(err)<0.25 else 0.33)
                w = k_w * err
            else:
                # Pure pursuit-like: drive to (tx,ty)
                dx, dy = tx - x, ty - y
                desired = math.atan2(dy, dx)
                err = wrap_pi(desired - heading)
                v_nom = v_max * max(0.35, 1.0 - min(1.0, abs(err)/1.7))
                v = v_nom
                w = k_w * err

            # Obstacle avoidance override
            if not reached_goal:
                if front < STOP_DIST:
                    v = 0
                    w = +1.3*k_w if left > right else -1.3*k_w
                elif front < SLOW_DIST:
                    v = min(v, v_slow)
                    # turn away from closer side
                    bias = (+0.8 if left > right else -0.8)
                    w = w + bias
            # Forward collision preview before move: do not enter obstacle
            nx = robot.x + v * math.cos(heading) * dt
            ny = robot.y + v * math.sin(heading) * dt
            if v > 0 and map2d.is_occupied(nx, ny):
                v = 0
            # Set wheel speeds (unicycle-to-dd)
            vl = v - 0.5*w*AXLE_LENGTH_PX
            vr = v + 0.5*w*AXLE_LENGTH_PX
            robot.set_wheels(vl, vr)

        # Move
        robot.kinematics(dt)
        positions.append([robot.x, robot.y])
        v_inst = (robot.vl + robot.vr) * 0.5
        speeds.append(abs(v_inst))
        # Check goal
        if not reached_goal and math.hypot(robot.x - goal_xy[0], robot.y - goal_xy[1]) <= 20.0:
            reached_goal = True
            robot.set_wheels(0, 0)
            if headless:
                break
        # Graphics and UI
        if not headless:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            gfx.clear()
            gfx.draw_points(path, color=(0,120,255), r=2, step=1)
            gfx.draw_robot(robot.x, robot.y, robot.heading)
            if "cloud" in sense:
                gfx.draw_sensor_data(sense["cloud"])
            status = f"Step:{step} Speed:{abs(v_inst):.1f} Goal:[{goal_xy[0]:.0f},{goal_xy[1]:.0f}]"
            gfx.draw_text(status, 10, 5)
            pygame.display.flip()
            clock.tick(60)
            if not running:
                break
    # Final UI for GUI: keep window until QUIT
    if not headless:
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
            pygame.display.flip()
            clock.tick(30)

    return {"positions": positions, "speeds": speeds, "collisions": collisions, "goal": [goal_xy[0], goal_xy[1]], "dt": dt}

def main():
    run_episode(max_steps=2000, dt=0.035, headless=False)

if __name__ == "__main__":
    main()
