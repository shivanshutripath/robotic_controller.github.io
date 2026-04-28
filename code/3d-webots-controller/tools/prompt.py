from string import Template

PROMPT = Template(r"""
Generate a Python controller for a Webots E-puck robot navigation task.

OUTPUT: A single Python file. No markdown. No explanations.

═══════════════════════════════════════════════════════════
REQUIRED FUNCTIONS (must be at module level)
═══════════════════════════════════════════════════════════

Use Supervisor() (not Robot) and read everything 
from the loaded world (no hard-coded obstacles/start/goal). 
In this world, motion is on the X-Y plane and height is Z. Find the goal as 
the Solid whose name is "GOAL" (also support DEF GOAL if present). Extract 
obstacles from WoodenBox (use its size field) and Rock (use its scale 
as radius/diameter approximation). Build an occupancy grid, run A* from robot 
pose to goal, then drive the e-puck along waypoints using real pose 
from the robot node translation+rotation (no dead-reckoning). 
Add reactive obstacle avoidance + recovery (back up + turn) to prevent 
getting stuck. Print + write run_result.json. Keep the controller alive 
at the end so extern doesn't disconnect. Output only the full code.

1. parse_world(world_text, params_text="") -> dict
   Parse the .wbt file and return:
   - "plane": "xy"
   - "start": {"x": float, "y": float} from E-puck translation
   - "goal": {"x": float, "y": float} from DEF GOAL translation  
   - "obstacles": list of {"x", "y", "sx", "sy"} from WoodenBox and Rock
   - "bounds": [x_min, x_max, y_min, y_max] containing start and goal with margin

2. build_grid(bounds, obstacles, resolution=0.1, inflation=0.0) -> list
   Return 2D occupancy grid (0=free, 1=occupied)

3. astar(grid, start_cell, goal_cell) -> list
   A* pathfinding, return list of (i,j) cells from start to goal

4. compute_wheel_speeds(pose_xy, yaw, waypoint_xy, prox) -> tuple
   Return (left_speed, right_speed) for differential drive toward waypoint.
   Use prox sensor readings to avoid collisions.

5. run_episode(...) -> dict
   Return {"success": bool, "steps": int}

6. main() -> None

End file with: if __name__ == "__main__": main()

═══════════════════════════════════════════════════════════
IMPORT RULES
═══════════════════════════════════════════════════════════

Top of file: import re, heapq, math
Inside run_episode/main ONLY: from controller import Supervisor

NEVER put Webots imports at module level.

═══════════════════════════════════════════════════════════
WORLD FILE
═══════════════════════════════════════════════════════════

$WORLD_WBT_TEXT

═══════════════════════════════════════════════════════════
REPAIR FEEDBACK
═══════════════════════════════════════════════════════════
$AUTO_REPAIR
""")