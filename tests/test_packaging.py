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


def _importlinter_config() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["importlinter"]


def test_every_module_lives_in_a_package():
    assert sorted(p.name for p in SRC.glob("*.py")) == []


def test_every_package_is_a_layer():
    """A package missing from the layers contract would go unchecked."""
    on_disk = {p.parent.name for p in SRC.glob("*/__init__.py")}
    config = _importlinter_config()
    top = next(c for c in config["contracts"] if c["type"] == "layers" and "containers" not in c)
    layered = {name.strip() for layer in top["layers"] for name in layer.split("|")}
    assert on_disk == set(config["root_packages"]) == layered


def test_src_is_the_import_root_not_a_package():
    assert not (SRC / "__init__.py").exists()
    for name in ("common.session_keys", "protocol"):
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


def test_the_baf_patch_imports_byok_from_its_package():
    """The vendored BAF file imports byok lazily and swallows the error, so a
    stale module path would silently turn BYOK off instead of failing."""
    tree = ast.parse((ROOT / "patches" / "websocket_platform.py").read_text(encoding="utf-8"))
    packages = {p.parent.name for p in SRC.glob("*/__init__.py")}
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module.split(".")[0] in packages:
            imported += [importlib.import_module(f"{node.module}.{alias.name}") for alias in node.names]
    assert imported and all(hasattr(m, "set_current") and hasattr(m, "reset_current") for m in imported)
