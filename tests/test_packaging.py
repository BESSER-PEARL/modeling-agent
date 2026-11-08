"""Packaging: src/ is the one import root, and pyproject.toml declares all of it.

The suite used to import the same code two ways (``from src.x`` and bare
``from x`` via sys.path inserts), which loads a module twice under two names.
"""
import ast
import importlib
import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _importlinter_config() -> dict:
    return _pyproject()["tool"]["importlinter"]


def _code_dirs() -> set:
    """Every directory under src/ that holds Python code."""
    return {path.parent for path in SRC.rglob("*.py")}


def test_every_module_lives_in_a_package():
    assert sorted(p.name for p in SRC.glob("*.py")) == []


def test_every_code_directory_is_a_package():
    """With namespaces = false, an install leaves out a directory without __init__.py."""
    missing = sorted(str(d.relative_to(SRC)) for d in _code_dirs() if not (d / "__init__.py").is_file())
    assert missing == []


def test_every_package_is_a_layer():
    """A package missing from the layers contract would go unchecked."""
    on_disk = {d.relative_to(SRC).parts[0] for d in _code_dirs() if d != SRC}
    config = _importlinter_config()
    top = next(c for c in config["contracts"] if c["type"] == "layers" and "containers" not in c)
    layered = {name.strip() for layer in top["layers"] for name in layer.split("|")}
    assert on_disk == set(config["root_packages"]) == layered


def test_src_is_the_import_root_not_a_package():
    assert not (SRC / "__init__.py").exists()
    module = importlib.import_module("common.session_keys")
    assert Path(module.__file__).resolve().is_relative_to(SRC)


def test_no_installed_package_shadows_ours():
    """The editable install puts src/ after site-packages, so a dependency that
    ships a top-level package of the same name (chromadb ships a schemas/
    directory) would be imported instead of ours."""
    shadowed = {}
    for name in _importlinter_config()["root_packages"]:
        origin = importlib.util.find_spec(name).origin
        if not (origin and Path(origin).resolve().is_relative_to(SRC)):
            shadowed[name] = origin
    assert shadowed == {}


def test_the_price_file_ships_with_its_package():
    """A non-editable install copies only declared package data: the price
    file sits in the package that reads it and matches its package-data."""
    from llm.tracking import token_tracker

    prices = token_tracker._PRICES_PATH
    relative = prices.relative_to(Path(token_tracker.__file__).parent)
    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["llm.tracking"]
    assert prices.is_file()
    assert any(relative.match(pattern) for pattern in patterns)


def test_the_user_metamodel_files_ship_with_their_package():
    """Same for the bundled user-profile metamodel and its semantics."""
    from utilities import user_metamodel

    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["utilities"]
    for path in (user_metamodel._METAMODEL_PATH, user_metamodel._SEMANTICS_PATH):
        relative = path.relative_to(Path(user_metamodel.__file__).parent)
        assert path.is_file()
        assert any(relative.match(pattern) for pattern in patterns)


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


def test_the_baf_patch_imports_resolve_in_the_agents_packages():
    """The vendored BAF file imports the agent's modules lazily and swallows
    the error, so a stale path would silently turn BYOK (or the memory
    heartbeat) off instead of failing."""
    tree = ast.parse((ROOT / "patches" / "websocket_platform.py").read_text(encoding="utf-8"))
    packages = {p.parent.name for p in SRC.glob("*/__init__.py")}
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module.split(".")[0] in packages:
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module}.{alias.name}"
                imported[f"{node.module}.{alias.name}"] = getattr(module, alias.name)
    byok = imported["llm.byok"]
    assert hasattr(byok, "set_current") and hasattr(byok, "reset_current")
    assert callable(imported["memory.touch_memory"])


def test_every_package_but_llm_is_kept_off_the_model_sdks():
    """A package missing from the SDK contract's hand-kept list could import
    openai / anthropic and bypass llm.provider unnoticed."""
    on_disk = {p.parent.name for p in SRC.glob("*/__init__.py")}
    contract = next(
        c for c in _importlinter_config()["contracts"]
        if c["name"] == "Only llm talks to the model SDKs"
    )
    assert set(contract["source_modules"]) == on_disk - {"llm"}
