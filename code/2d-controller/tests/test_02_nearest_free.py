# tests/test_02_nearest_free.py
import numpy as np
import pytest

@pytest.mark.runtime
def test_nearest_free_returns_free(controller_module):
    assert hasattr(controller_module, "nearest_free"), "controller.py must define nearest_free(map2d,x,y,...)"

    OccMap2D = controller_module.OccMap2D
    occ = np.zeros((50, 50), dtype=bool)
    # make a 10x10 obstacle block
    occ[20:30, 20:30] = True
    m = OccMap2D(occ)

    x, y = controller_module.nearest_free(m, 25, 25, max_r=50)
    assert m.is_occupied(x, y) is False
