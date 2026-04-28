# tests/test_03_planner_api_and_validity.py
import numpy as np
import pytest

PLANNER_CANDIDATES = ["plan_path", "astar_plan", "a_star_plan", "astar", "a_star"]

@pytest.mark.runtime
def test_planner_function_exists(controller_module):
    found = [n for n in PLANNER_CANDIDATES if hasattr(controller_module, n)]
    assert found, f"Missing planner function. Expected one of: {PLANNER_CANDIDATES}"

@pytest.mark.runtime
def test_plan_path_returns_pixel_path(controller_module):
    fn = next(getattr(controller_module, n) for n in PLANNER_CANDIDATES if hasattr(controller_module, n))
    OccMap2D = controller_module.OccMap2D

    occ = np.zeros((80, 80), dtype=bool)
    # vertical wall with a gap
    occ[:, 40] = True
    occ[35:45, 40] = False

    m = OccMap2D(occ)
    cell = 4
    start = (10, 10)
    goal = (70, 70)

    path = fn(m, start, goal, cell)

    assert isinstance(path, list)
    assert len(path) >= 2, "Planner returned empty or trivial path"
    assert all(isinstance(p, (tuple, list)) and len(p) == 2 for p in path)

    # Must be Python ints (not np.int64)
    for x, y in path[:: max(1, len(path)//20)]:
        assert isinstance(x, int) and isinstance(y, int), "Path points must be Python ints"

    # All points must be in free space
    for x, y in path[:: max(1, len(path)//20)]:
        assert m.is_occupied(x, y) is False

    # Endpoints should be in the vicinity of start/goal (not necessarily equal)
    x0, y0 = path[0]
    x1, y1 = path[-1]
    assert (x0 - start[0])**2 + (y0 - start[1])**2 <= (3*cell)**2
    assert (x1 - goal[0])**2 + (y1 - goal[1])**2 <= (3*cell)**2

    # Avoid degenerate path of identical points
    unique = set((x, y) for x, y in path)
    assert len(unique) >= 2, "Degenerate path: points are not changing"
