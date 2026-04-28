# tests/test_04_episode_schema.py
import pytest

EPISODE_FN_CANDIDATES = ["run_episode", "simulate_episode"]

@pytest.mark.e2e
def test_episode_schema(controller_module):
    found = [n for n in EPISODE_FN_CANDIDATES if hasattr(controller_module, n)]
    assert found, "Need run_episode(...) or simulate_episode(...)"
    run_episode = getattr(controller_module, found[0])

    out = run_episode(max_steps=50, dt=0.05, start_xy=None, goal_xy=None, headless=True)

    for k in ["positions","speeds","collisions","goal","dt"]:
        assert k in out, f"Missing key: {k}"

    assert out["goal"] is not None, "goal must never be None"
    assert isinstance(out["positions"], list) and len(out["positions"]) > 0, "positions must be non-empty"
    assert len(out["positions"]) == len(out["speeds"]), "positions and speeds length mismatch"
