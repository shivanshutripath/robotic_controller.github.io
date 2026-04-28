"""Test grid building."""


def test_build_grid_exists(controller_module):
    assert hasattr(controller_module, "build_grid")


def test_returns_2d_grid(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    assert len(grid) > 0
    assert len(grid[0]) > 0


def test_grid_dimensions_reasonable(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    x_min, x_max, y_min, y_max = out["bounds"]
    expected_w = int((x_max - x_min) / 0.1) + 1
    expected_h = int((y_max - y_min) / 0.1) + 1
    
    assert len(grid) >= expected_w - 5
    assert len(grid[0]) >= expected_h - 5


def test_marks_obstacles(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    grid = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    
    occupied = sum(sum(1 for c in row if c != 0) for row in grid)
    assert occupied > 0, "Grid must mark obstacle cells"


def test_inflation_increases_occupied_cells(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    
    grid_no = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.0)
    grid_yes = controller_module.build_grid(out["bounds"], out["obstacles"], 0.1, 0.15)
    
    occ_no = sum(sum(1 for c in row if c != 0) for row in grid_no)
    occ_yes = sum(sum(1 for c in row if c != 0) for row in grid_yes)
    
    assert occ_yes >= occ_no