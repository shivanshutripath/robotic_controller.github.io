# tests/test_01_occmap2d_safety.py
import numpy as np
import pytest

@pytest.mark.runtime
def test_occmap2d_is_safe(controller_module):
    OccMap2D = getattr(controller_module, "OccMap2D", None)
    assert OccMap2D is not None, "controller.py must define OccMap2D"

    occ = np.zeros((10, 12), dtype=bool)
    m = OccMap2D(occ)

    assert m.is_occupied(12, 0) is True
    assert m.is_occupied(0, 10) is True
    assert m.is_occupied(-1, 0) is True
    assert m.is_occupied(0, -1) is True
    assert m.is_occupied(0, 0) in (True, False)
