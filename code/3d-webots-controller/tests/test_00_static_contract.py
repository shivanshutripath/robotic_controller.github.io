import ast
import re
import pytest

FORBIDDEN_TOPLEVEL_MODULES = {"controller"}
FORBIDDEN_LIBS = {"numpy", "pandas", "torch", "tensorflow", "cv2"}


def test_file_is_valid_python(controller_path):
    """Controller must be syntactically valid Python."""
    src = controller_path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"Syntax error: {e}")


def test_no_markdown_fences(controller_path):
    """No markdown code fences allowed."""
    src = controller_path.read_text(encoding="utf-8")
    assert "```" not in src


def test_has_main_guard(controller_path):
    """Must have if __name__ == '__main__' guard."""
    src = controller_path.read_text(encoding="utf-8")
    assert re.search(r'__name__\s*==\s*[\'"]__main__[\'"]', src)


def test_required_functions_exist(controller_module):
    """All 6 required functions must exist."""
    required = ["parse_world", "build_grid", "astar", "compute_wheel_speeds", "run_episode", "main"]
    for name in required:
        assert hasattr(controller_module, name), f"Missing function: {name}()"


def test_no_forbidden_imports(controller_path):
    """No numpy, pandas, tensorflow, etc."""
    src = controller_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = set()
    
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in FORBIDDEN_LIBS:
                    bad.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            if n.module.split(".")[0] in FORBIDDEN_LIBS:
                bad.add(n.module.split(".")[0])
    
    assert not bad, f"Forbidden imports: {sorted(bad)}"


def test_webots_import_not_at_top_level(controller_path):
    """CRITICAL: Webots imports must be inside functions only."""
    src = controller_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "controller":
            pytest.fail(f"Line {node.lineno}: Top-level 'from controller import' FORBIDDEN")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in FORBIDDEN_TOPLEVEL_MODULES:
                    pytest.fail(f"Line {node.lineno}: Top-level 'import controller' FORBIDDEN")