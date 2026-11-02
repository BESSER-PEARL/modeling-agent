"""Packaging: src/ is the one import root, and pyproject.toml declares all of it.

The suite used to import the same code two ways (``from src.x`` and bare
``from x`` via sys.path inserts), which loads a module twice under two names.
"""
import ast
import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _setuptools_config() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]


def test_every_top_level_module_is_declared():
    on_disk = {p.stem for p in SRC.glob("*.py")}
    assert on_disk == set(_setuptools_config()["py-modules"])


def test_src_is_the_import_root_not_a_package():
    assert not (SRC / "__init__.py").exists()
    for name in ("session_keys", "protocol"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(SRC)


def test_nothing_imports_through_a_src_prefix():
    offenders = []
    files = [*SRC.rglob("*.py"), *(ROOT / "tests").rglob("*.py"), *(ROOT / "patches").glob("*.py"), ROOT / "modeling_agent.py"]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            offenders += [f"{path.relative_to(ROOT)}:{node.lineno}" for n in names if n == "src" or n.startswith("src.")]
    assert offenders == []
