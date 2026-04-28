import pytest
import inspect

EPISODE_FN_CANDIDATES = ("run_episode", "simulate_episode")

@pytest.mark.contract
def test_nearest_free_signature(controller_module):
    assert hasattr(controller_module, "nearest_free"), "nearest_free missing"
    sig = inspect.signature(controller_module.nearest_free)
    assert "max_r" in sig.parameters, "nearest_free must accept keyword max_r"
    # ensure keyword call works
    OccMap2D = controller_module.OccMap2D
    import numpy as np
    m = OccMap2D(np.zeros((10, 10), dtype=bool))
    x, y = controller_module.nearest_free(m, 5, 5, max_r=3)
    assert isinstance(x, int) and isinstance(y, int)

@pytest.mark.contract
def test_run_episode_never_raises(controller_module):
    found = [n for n in EPISODE_FN_CANDIDATES if hasattr(controller_module, n)]
    assert found, "Need run_episode(...) or simulate_episode(...)"
    fn = getattr(controller_module, found[0])
    out = fn(max_steps=10, dt=0.05, start_xy=None, goal_xy=None, headless=True)
    assert isinstance(out, dict)
    for k in ("positions", "speeds", "collisions", "goal", "dt"):
        assert k in out

@pytest.mark.contract
def test_no_scipy(controller_source):
    assert "scipy" not in controller_source.lower(), "scipy is forbidden (not installed)"

@pytest.mark.contract
def test_no_raise_runtimeerror(controller_source):
    assert "raise RuntimeError" not in controller_source, "Do not raise RuntimeError in controller"
