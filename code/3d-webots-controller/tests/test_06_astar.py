"""Test A* pathfinding."""


def _world_to_grid(x, y, bounds, res):
    x_min, x_max, y_min, y_max = bounds
    return (int((x - x_min) / res), int((y - y_min) / res))


def _clear_area(grid, cell, radius=2):
    """Helper to clear area around cell."""
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            x, y = cell[0] + dx, cell[1] + dy
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                grid[x][y] = 0


def test_astar_exists(controller_module):
    assert hasattr(controller_module, "astar")


def test_returns_list(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    s = _world_to_grid(out["start"]["x"], out["start"]["y"], out["bounds"], 0.1)
    g = _world_to_grid(out["goal"]["x"], out["goal"]["y"], out["bounds"], 0.1)
    
    _clear_area(grid, s)
    _clear_area(grid, g)
    
    path = controller_module.astar(grid, s, g)
    assert isinstance(path, list)


def test_finds_non_empty_path(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    s = _world_to_grid(out["start"]["x"], out["start"]["y"], out["bounds"], 0.1)
    g = _world_to_grid(out["goal"]["x"], out["goal"]["y"], out["bounds"], 0.1)
    
    _clear_area(grid, s)
    _clear_area(grid, g)
    
    path = controller_module.astar(grid, s, g)
    assert len(path) > 0


def test_path_starts_at_start(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    s = _world_to_grid(out["start"]["x"], out["start"]["y"], out["bounds"], 0.1)
    g = _world_to_grid(out["goal"]["x"], out["goal"]["y"], out["bounds"], 0.1)
    
    _clear_area(grid, s)
    _clear_area(grid, g)
    
    path = controller_module.astar(grid, s, g)
    assert path[0] == s


def test_path_ends_at_goal(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    s = _world_to_grid(out["start"]["x"], out["start"]["y"], out["bounds"], 0.1)
    g = _world_to_grid(out["goal"]["x"], out["goal"]["y"], out["bounds"], 0.1)
    
    _clear_area(grid, s)
    _clear_area(grid, g)
    
    path = controller_module.astar(grid, s, g)
    assert path[-1] == g


def test_path_cells_in_bounds(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    s = _world_to_grid(out["start"]["x"], out["start"]["y"], out["bounds"], 0.1)
    g = _world_to_grid(out["goal"]["x"], out["goal"]["y"], out["bounds"], 0.1)
    
    _clear_area(grid, s)
    _clear_area(grid, g)
    
    path = controller_module.astar(grid, s, g)
    for x, y in path:
        assert 0 <= x < len(grid), f"Path x={x} out of bounds"
        assert 0 <= y < len(grid[0]), f"Path y={y} out of bounds"


def test_path_avoids_obstacles(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    s = _world_to_grid(out["start"]["x"], out["start"]["y"], out["bounds"], 0.1)
    g = _world_to_grid(out["goal"]["x"], out["goal"]["y"], out["bounds"], 0.1)
    
    _clear_area(grid, s)
    _clear_area(grid, g)
    
    path = controller_module.astar(grid, s, g)
    for x, y in path:
        assert grid[x][y] == 0, f"Path goes through obstacle at ({x}, {y})"


def test_path_is_connected(controller_module, world_text, params_text):
    """Each step must be adjacent to previous (8-connectivity)."""
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    s = _world_to_grid(out["start"]["x"], out["start"]["y"], out["bounds"], 0.1)
    g = _world_to_grid(out["goal"]["x"], out["goal"]["y"], out["bounds"], 0.1)
    
    _clear_area(grid, s)
    _clear_area(grid, g)
    
    path = controller_module.astar(grid, s, g)
    for i in range(len(path) - 1):
        dx = abs(path[i+1][0] - path[i][0])
        dy = abs(path[i+1][1] - path[i][1])
        assert dx <= 1 and dy <= 1, f"Path not connected at step {i}"