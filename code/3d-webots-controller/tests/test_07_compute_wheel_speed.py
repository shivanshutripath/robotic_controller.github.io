"""Test wheel speed computation (control logic)."""

import inspect


def test_compute_wheel_speeds_exists(controller_module):
    assert hasattr(controller_module, "compute_wheel_speeds")


def test_has_correct_signature(controller_module):
    sig = inspect.signature(controller_module.compute_wheel_speeds)
    params = list(sig.parameters.keys())
    assert len(params) == 4, f"Expected 4 params, got {len(params)}"


def test_returns_tuple_of_two(controller_module):
    result = controller_module.compute_wheel_speeds(
        pose_xy=(0, 0), yaw=0, waypoint_xy=(1, 0), prox=[0]*8
    )
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_forward_motion_when_aligned(controller_module):
    """Waypoint straight ahead → both wheels forward."""
    left, right = controller_module.compute_wheel_speeds(
        pose_xy=(0.0, 0.0),
        yaw=0.0,
        waypoint_xy=(1.0, 0.0),
        prox=[0] * 8
    )
    assert left > 0 and right > 0


def test_turns_left_when_waypoint_left(controller_module):
    """Waypoint to the left → right wheel faster."""
    left, right = controller_module.compute_wheel_speeds(
        pose_xy=(0.0, 0.0),
        yaw=0.0,
        waypoint_xy=(0.0, 1.0),
        prox=[0] * 8
    )
    assert right > left


def test_turns_right_when_waypoint_right(controller_module):
    """Waypoint to the right → left wheel faster."""
    left, right = controller_module.compute_wheel_speeds(
        pose_xy=(0.0, 0.0),
        yaw=0.0,
        waypoint_xy=(0.0, -1.0),
        prox=[0] * 8
    )
    assert left > right


def test_recovery_when_front_blocked(controller_module):
    """High front proximity → backup or turn."""
    prox = [200, 200, 0, 0, 0, 0, 200, 200]
    left, right = controller_module.compute_wheel_speeds(
        pose_xy=(0.0, 0.0),
        yaw=0.0,
        waypoint_xy=(1.0, 0.0),
        prox=prox
    )
    # Either backing up or turning in place
    assert (left < 0 and right < 0) or (left * right < 0)