# tests/test_05_episode_progress.py
import math
import pytest

EPISODE_FN_CANDIDATES = ["run_episode", "simulate_episode"]

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

@pytest.mark.e2e
def test_episode_makes_progress(controller_module):
    run_episode = getattr(controller_module, next(n for n in EPISODE_FN_CANDIDATES if hasattr(controller_module, n)))
    out = run_episode(max_steps=400, dt=0.05, start_xy=None, goal_xy=None, headless=True)

    goal = tuple(out["goal"])
    positions = [tuple(p) for p in out["positions"]]

    d0 = dist(positions[0], goal)
    dmin = min(dist(p, goal) for p in positions)

    assert d0 < float("inf")
    assert dmin <= 0.7 * d0, f"No real progress toward goal: d0={d0:.1f}, dmin={dmin:.1f}"
