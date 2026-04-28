# tests/conftest.py
import os
import sys
import importlib
import pytest

@pytest.fixture(scope="session", autouse=True)
def _pygame_headless():
    # must be set BEFORE pygame import inside controller
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    yield

@pytest.fixture(scope="session")
def project_root():
    # repository root = one level above tests/
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[1])

@pytest.fixture()
def controller_module(project_root, monkeypatch):
    # Ensure root on sys.path so "import controller" works
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import fresh each time (important when controller.py regenerated)
    if "controller" in sys.modules:
        mod = importlib.reload(sys.modules["controller"])
    else:
        mod = importlib.import_module("controller")
    return mod
