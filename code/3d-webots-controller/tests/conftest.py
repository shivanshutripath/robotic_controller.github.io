import os
import importlib.util
from pathlib import Path
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root()


@pytest.fixture(scope="session")
def world_path(repo_root: Path) -> Path:
    p = os.environ.get("WORLD_PATH", "").strip()
    path = Path(p).resolve() if p else (repo_root / "worlds" / "empty.wbt").resolve()
    assert path.exists(), f"World not found: {path}"
    return path


@pytest.fixture(scope="session")
def world_text(world_path: Path) -> str:
    return world_path.read_text(encoding="utf-8", errors="ignore")


@pytest.fixture(scope="session")
def controller_path(repo_root: Path) -> Path:
    p = os.environ.get("CONTROLLER_PATH", "").strip()
    path = Path(p).resolve() if p else (repo_root / "controllers" / "obs_avoidance" / "obs_avoidance.py").resolve()
    assert path.exists(), f"Controller not found: {path}"
    return path


@pytest.fixture(scope="session")
def params_text() -> str:
    p = os.environ.get("PARAMS_PATH", "").strip()
    if not p:
        return ""
    path = Path(p).resolve()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


@pytest.fixture()
def controller_module(controller_path: Path):
    spec = importlib.util.spec_from_file_location("gen_controller", str(controller_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod