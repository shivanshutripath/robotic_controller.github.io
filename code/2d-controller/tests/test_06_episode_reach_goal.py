# tests/test_06_episode_reach_goal.py
import pytest
from tests.metrics import compute_metrics   # if you made tests a package; else: from metrics import compute_metrics

EPISODE_FN_CANDIDATES = ["run_episode", "simulate_episode"]

@pytest.mark.e2e
def test_episode_reaches_goal(controller_module):
    run_episode = getattr(controller_module, next(n for n in EPISODE_FN_CANDIDATES if hasattr(controller_module, n)))

    out = run_episode(max_steps=2500, dt=0.05, start_xy=None, goal_xy=None, headless=True)

    positions = [tuple(p) for p in out["positions"]]
    speeds = list(out["speeds"])
    collisions = int(out["collisions"])
    goal = tuple(out["goal"])
    dt = float(out["dt"])

    m = compute_metrics(positions, speeds, collisions, goal, dt, goal_radius=20.0)

    assert collisions == 0, f"Collisions: {collisions}"
    assert m.reached_goal, f"Did not reach goal. min_goal_dist={m.min_goal_dist:.2f}"
