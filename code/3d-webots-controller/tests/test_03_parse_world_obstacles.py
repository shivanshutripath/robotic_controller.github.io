"""Test obstacle parsing."""

import math


def test_minimum_obstacle_count(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    # 3 WoodenBox + 2 Rock = 5 minimum
    assert len(out["obstacles"]) >= 5


def test_each_obstacle_has_required_fields(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    for i, o in enumerate(out["obstacles"]):
        assert isinstance(o, dict), f"Obstacle {i} not dict"
        for k in ("x", "y", "sx", "sy"):
            assert k in o, f"Obstacle {i} missing {k}"
            assert math.isfinite(o[k]), f"Obstacle {i}.{k} not finite"


def test_obstacle_sizes_positive(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    for i, o in enumerate(out["obstacles"]):
        assert o["sx"] > 0, f"Obstacle {i}: sx not positive"
        assert o["sy"] > 0, f"Obstacle {i}: sy not positive"


def test_obstacle_sizes_reasonable(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    for i, o in enumerate(out["obstacles"]):
        assert 0.01 < o["sx"] < 5.0, f"Obstacle {i}: sx unreasonable"
        assert 0.01 < o["sy"] < 5.0, f"Obstacle {i}: sy unreasonable"


def test_specific_wooden_box_found(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    # WoodenBox at 0.194803, 0.0218687, size 0.2 x 0.2
    found = any(
        abs(o["x"] - 0.194803) < 0.03 and
        abs(o["y"] - 0.0218687) < 0.03 and
        abs(o["sx"] - 0.2) < 0.03 and
        abs(o["sy"] - 0.2) < 0.03
        for o in out["obstacles"]
    )
    assert found, "Expected WoodenBox not found"


def test_obstacles_within_bounds(controller_module, world_text, params_text):
    out = controller_module.parse_world(world_text, params_text)
    x_min, x_max, y_min, y_max = out["bounds"]
    margin = 0.5
    for i, o in enumerate(out["obstacles"]):
        assert x_min - margin <= o["x"] <= x_max + margin, f"Obstacle {i} x out of bounds"
        assert y_min - margin <= o["y"] <= y_max + margin, f"Obstacle {i} y out of bounds"