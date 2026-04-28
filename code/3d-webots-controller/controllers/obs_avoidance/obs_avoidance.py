import re
import heapq
import math

def parse_world(world_text, params_text=""):
    result = {
        "plane": "xy",
        "start": {"x": 0.0, "y": 0.0},
        "goal": {"x": 0.0, "y": 0.0},
        "obstacles": [],
        "bounds": [-1.0, 1.0, -1.0, 1.0]
    }
    
    lines = world_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if 'E-puck' in line and '{' in line:
            j = i + 1
            brace_count = 1
            while j < len(lines) and brace_count > 0:
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                if 'translation' in lines[j] and 'hidden' not in lines[j]:
                    match = re.search(r'translation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', lines[j])
                    if match:
                        result["start"]["x"] = float(match.group(1))
                        result["start"]["y"] = float(match.group(2))
                j += 1
        
        if 'DEF GOAL' in line or ('Solid' in line and 'GOAL' in line):
            j = i + 1
            brace_count = 1
            while j < len(lines) and brace_count > 0:
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                if 'translation' in lines[j]:
                    match = re.search(r'translation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', lines[j])
                    if match:
                        result["goal"]["x"] = float(match.group(1))
                        result["goal"]["y"] = float(match.group(2))
                j += 1
        
        if 'WoodenBox' in line and '{' in line:
            j = i + 1
            brace_count = 1
            obs = {"x": 0.0, "y": 0.0, "sx": 0.1, "sy": 0.1}
            while j < len(lines) and brace_count > 0:
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                if 'translation' in lines[j]:
                    match = re.search(r'translation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', lines[j])
                    if match:
                        obs["x"] = float(match.group(1))
                        obs["y"] = float(match.group(2))
                if 'size' in lines[j]:
                    match = re.search(r'size\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', lines[j])
                    if match:
                        obs["sx"] = float(match.group(1))
                        obs["sy"] = float(match.group(2))
                j += 1
            result["obstacles"].append(obs)
        
        if 'Rock' in line and '{' in line:
            j = i + 1
            brace_count = 1
            obs = {"x": 0.0, "y": 0.0, "sx": 0.1, "sy": 0.1}
            scale = 1.0
            while j < len(lines) and brace_count > 0:
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                if 'translation' in lines[j]:
                    match = re.search(r'translation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', lines[j])
                    if match:
                        obs["x"] = float(match.group(1))
                        obs["y"] = float(match.group(2))
                if 'scale' in lines[j]:
                    match = re.search(r'scale\s+([-\d.e]+)', lines[j])
                    if match:
                        scale = float(match.group(1))
                j += 1
            radius = 0.1 * scale
            obs["sx"] = radius * 2
            obs["sy"] = radius * 2
            result["obstacles"].append(obs)
        
        i += 1
    
    all_x = [result["start"]["x"], result["goal"]["x"]]
    all_y = [result["start"]["y"], result["goal"]["y"]]
    for obs in result["obstacles"]:
        all_x.extend([obs["x"] - obs["sx"]/2, obs["x"] + obs["sx"]/2])
        all_y.extend([obs["y"] - obs["sy"]/2, obs["y"] + obs["sy"]/2])
    
    margin = 0.2
    result["bounds"] = [min(all_x) - margin, max(all_x) + margin, min(all_y) - margin, max(all_y) + margin]
    
    return result

def build_grid(bounds, obstacles, resolution=0.1, inflation=0.05):
    x_min, x_max, y_min, y_max = bounds
    cols = max(1, int(math.ceil((x_max - x_min) / resolution)))
    rows = max(1, int(math.ceil((y_max - y_min) / resolution)))
    
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for obs in obstacles:
        ox, oy = obs["x"], obs["y"]
        sx, sy = obs["sx"] / 2 + inflation, obs["sy"] / 2 + inflation
        
        for r in range(rows):
            for c in range(cols):
                wx = x_min + c * resolution + resolution / 2
                wy = y_min + r * resolution + resolution / 2
                
                if abs(wx - ox) <= sx and abs(wy - oy) <= sy:
                    grid[r][c] = 1
    
    return grid

def astar(grid, start_cell, goal_cell):
    rows = len(grid)
    cols = len(grid[0])
    
    def heuristic(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    open_set = []
    heapq.heappush(open_set, (0, start_cell))
    came_from = {}
    g_score = {start_cell: 0}
    f_score = {start_cell: heuristic(start_cell, goal_cell)}
    
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
                
                move_cost = math.sqrt(dr*dr + dc*dc)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []

def compute_wheel_speeds(pose_xy, yaw, waypoint_xy, prox):
    max_speed = 6.28
    
    dx = waypoint_xy[0] - pose_xy[0]
    dy = waypoint_xy[1] - pose_xy[1]
    target_angle = math.atan2(dy, dx)
    
    angle_error = target_angle - yaw
    while angle_error > math.pi:
        angle_error -= 2 * math.pi
    while angle_error < -math.pi:
        angle_error += 2 * math.pi
    
    front_sensors = prox[0:3] + prox[5:8] if len(prox) >= 8 else prox
    front_left = prox[0:3] if len(prox) >= 3 else []
    front_right = prox[5:8] if len(prox) >= 8 else []
    
    obstacle_threshold = 80
    
    left_obs = any(p > obstacle_threshold for p in front_left) if front_left else False
    right_obs = any(p > obstacle_threshold for p in front_right) if front_right else False
    front_obs = any(p > obstacle_threshold for p in front_sensors) if front_sensors else False
    
    if front_obs:
        if left_obs and not right_obs:
            return max_speed * 0.5, -max_speed * 0.5
        elif right_obs and not left_obs:
            return -max_speed * 0.5, max_speed * 0.5
        else:
            return -max_speed * 0.3, max_speed * 0.3
    
    if abs(angle_error) > 0.3:
        turn_speed = max_speed * 0.5 * (1 if angle_error > 0 else -1)
        return -turn_speed, turn_speed
    
    forward_speed = max_speed * 0.8
    turn_factor = angle_error * 2.0
    
    left_speed = forward_speed - turn_factor * max_speed * 0.3
    right_speed = forward_speed + turn_factor * max_speed * 0.3
    
    left_speed = max(-max_speed, min(max_speed, left_speed))
    right_speed = max(-max_speed, min(max_speed, right_speed))
    
    return left_speed, right_speed

def run_episode(world_data, max_steps=2000):
    from controller import Supervisor
    
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    
    robot_node = supervisor.getFromDef("E-puck")
    if robot_node is None:
        robot_node = supervisor.getSelf()
    
    goal_node = supervisor.getFromDef("GOAL")
    if goal_node is None:
        root = supervisor.getRoot()
        children = root.getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            if node.getTypeName() == "Solid":
                name_field = node.getField("name")
                if name_field and name_field.getSFString() == "GOAL":
                    goal_node = node
                    break
    
    left_motor = supervisor.getDevice("left wheel motor")
    right_motor = supervisor.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    
    ps_sensors = []
    for i in range(8):
        ps = supervisor.getDevice(f"ps{i}")
        ps.enable(timestep)
        ps_sensors.append(ps)
    
    bounds = world_data["bounds"]
    obstacles = world_data["obstacles"]
    resolution = 0.05
    
    grid = build_grid(bounds, obstacles, resolution, inflation=0.08)
    
    def world_to_cell(x, y):
        c = int((x - bounds[0]) / resolution)
        r = int((y - bounds[2]) / resolution)
        r = max(0, min(len(grid) - 1, r))
        c = max(0, min(len(grid[0]) - 1, c))
        return (r, c)
    
    def cell_to_world(r, c):
        x = bounds[0] + c * resolution + resolution / 2
        y = bounds[2] + r * resolution + resolution / 2
        return (x, y)
    
    goal_x = world_data["goal"]["x"]
    goal_y = world_data["goal"]["y"]
    goal_threshold = 0.08
    
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")
    
    pos = trans_field.getSFVec3f()
    start_x, start_y = pos[0], pos[1]
    
    start_cell = world_to_cell(start_x, start_y)
    goal_cell = world_to_cell(goal_x, goal_y)
    
    path = astar(grid, start_cell, goal_cell)
    
    if not path:
        print("No path found!")
        return {"success": False, "steps": 0}
    
    waypoints = [cell_to_world(r, c) for r, c in path]
    
    skip = max(1, len(waypoints) // 20)
    waypoints = waypoints[::skip] + [waypoints[-1]]
    
    current_waypoint_idx = 0
    steps = 0
    stuck_counter = 0
    last_pos = (start_x, start_y)
    
    while supervisor.step(timestep) != -1 and steps < max_steps:
        steps += 1
        
        pos = trans_field.getSFVec3f()
        rot = rot_field.getSFRotation()
        
        x, y = pos[0], pos[1]
        
        axis_z = rot[2]
        angle = rot[3]
        yaw = angle if axis_z >= 0 else -angle
        
        dist_to_goal = math.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        if dist_to_goal < goal_threshold:
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)
            print(f"Goal reached in {steps} steps!")
            return {"success": True, "steps": steps}
        
        if current_waypoint_idx < len(waypoints):
            wp = waypoints[current_waypoint_idx]
            dist_to_wp = math.sqrt((x - wp[0])**2 + (y - wp[1])**2)
            if dist_to_wp < 0.1:
                current_waypoint_idx += 1
        
        if current_waypoint_idx >= len(waypoints):
            target = (goal_x, goal_y)
        else:
            target = waypoints[current_waypoint_idx]
        
        prox = [ps.getValue() for ps in ps_sensors]
        
        move_dist = math.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
        if move_dist < 0.001:
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_pos = (x, y)
        
        if stuck_counter > 50:
            left_motor.setVelocity(-3.0)
            right_motor.setVelocity(-2.0)
            stuck_counter = 0
            continue
        
        left_speed, right_speed = compute_wheel_speeds((x, y), yaw, target, prox)
        
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
    
    print(f"Max steps reached: {steps}")
    return {"success": False, "steps": steps}

def main():
    import json
    
    world_file_path = "../../worlds/obs_avoidance.wbt"
    try:
        with open(world_file_path, 'r') as f:
            world_text = f.read()
    except:
        world_text = ""
    
    world_data = parse_world(world_text)
    print(f"Start: {world_data['start']}")
    print(f"Goal: {world_data['goal']}")
    print(f"Obstacles: {len(world_data['obstacles'])}")
    
    result = run_episode(world_data)
    
    print(f"Result: {result}")
    
    with open("run_result.json", "w") as f:
        json.dump(result, f)
    
    from controller import Supervisor
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    while supervisor.step(timestep) != -1:
        pass

if __name__ == "__main__":
    main()
