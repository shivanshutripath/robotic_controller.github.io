"""Test parse_world function contract."""

import math


def test_parse_world_exists(controller_module):
    assert hasattr(controller_module, "parse_world")


def test_parse_world_returns_dict(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    assert isinstance(out, dict)


def test_has_all_required_keys(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    for k in ["plane", "start", "goal", "obstacles", "bounds"]:
        assert k in out, f"Missing key: {k}"


def test_plane_is_xy(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    plane = (out.get("plane") or "xy").lower().replace("-", "")
    assert plane == "xy"


def test_start_structure(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    s = out["start"]
    assert isinstance(s, dict)
    assert "x" in s and "y" in s
    assert math.isfinite(s["x"]) and math.isfinite(s["y"])


def test_goal_structure(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    g = out["goal"]
    assert g is not None
    assert isinstance(g, dict)
    assert "x" in g and "y" in g
    assert math.isfinite(g["x"]) and math.isfinite(g["y"])


def test_obstacles_is_list(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    assert isinstance(out["obstacles"], list)


def test_bounds_is_4_element_list(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    b = out["bounds"]
    assert isinstance(b, (list, tuple))
    assert len(b) == 4


def test_bounds_are_valid_ranges(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    x_min, x_max, y_min, y_max = out["bounds"]
    assert x_min < x_max
    assert y_min < y_max


def test_all_bounds_finite(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    assert all(math.isfinite(v) for v in out["bounds"])