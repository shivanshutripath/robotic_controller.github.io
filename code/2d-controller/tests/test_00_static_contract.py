# tests/test_00_contract_static.py
import ast
import re
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def controller_source(project_root):
    p = Path(project_root) / "controller.py"
    return p.read_text(encoding="utf-8")

@pytest.mark.contract
def test_has_main_and_guard(controller_source):
    assert "def main" in controller_source
    assert 'if __name__ == "__main__":' in controller_source or "if __name__ == '__main__':" in controller_source

@pytest.mark.contract
def test_allowed_imports_only(controller_source):
    allowed = {"math","heapq","json","pygame","numpy","os","time","robot"}
    tree = ast.parse(controller_source)

    # no imports inside defs/classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    raise AssertionError("Imports inside functions/classes are forbidden")

    # only allowed modules
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                root = n.name.split(".")[0]
                assert root in allowed, f"Forbidden import: {n.name}"
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root in allowed, f"Forbidden import-from: {node.module}"

@pytest.mark.contract
def test_required_constants_present(controller_source):
    for name in [
        "MAP_IMG","PARAMS_JSON","ROBOT_IMG","AXLE_LENGTH_PX","SENSOR_RANGE_PX",
        "SENSOR_FOV_DEG","N_RAYS","LOOKAHEAD","STOP_DIST","SLOW_DIST"
    ]:
        assert re.search(rf"^\s*{name}\s*=", controller_source, re.M), f"Missing constant: {name}"

@pytest.mark.contract
def test_no_surface_indexing(controller_source):
    # Ban pygame.Surface indexing; allow numpy arr[...,0]
    forbidden = [
        r"gfx\.map_img\s*\[",                 # gfx.map_img[...]
        r"\b(surface|surf|img)\s*\[",         # surface[...] variable patterns
    ]
    for pat in forbidden:
        assert not re.search(pat, controller_source), f"Forbidden Surface indexing: {pat}"
