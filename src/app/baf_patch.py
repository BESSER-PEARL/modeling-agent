"""Install the vendored BAF WebSocket platform (``patches/websocket_platform.py``).

The Docker image copies the patched file over the pip-installed one at build
time. A run outside Docker (local dev, on-prem) would otherwise use stock BAF:
the BYOK key is logged in plain text and ignored, and there is no reply outbox
or ownership-guarded slot reclaim. ``install()`` loads the vendored module under
BAF's own module name before ``baf.core.agent`` imports it, so both paths run
the same code; ``verify()`` refuses to start if that did not take.

The vendored file is a copy of one BAF release's module, so ``install()`` first
checks that the installed BAF is the version pinned in ``requirements.txt``.
"""

import importlib
import importlib.util
import os
import re
import sys
from importlib import metadata

_MODULE = "baf.platforms.websocket.websocket_platform"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATCH_FILE = os.path.join(_ROOT, "patches", "websocket_platform.py")
_REQUIREMENTS = os.path.join(_ROOT, "requirements.txt")
_BAF_DIST = "besser-agentic-framework"
_PIN_RE = re.compile(rf"^{_BAF_DIST}(?:\[[^\]]*\])?\s*==\s*([^\s#;]+)", re.MULTILINE)


def _is_patched(platform_cls) -> bool:
    return platform_cls is not None and hasattr(platform_cls, "_flush_outbox")


def pinned_baf_version() -> str:
    """The BAF version ``requirements.txt`` pins (the one the patch was vendored from)."""
    with open(_REQUIREMENTS, encoding="utf-8") as handle:
        match = _PIN_RE.search(handle.read())
    if match is None:
        raise RuntimeError(f"No '{_BAF_DIST} == <version>' pin in {_REQUIREMENTS}")
    return match.group(1)


def check_baf_version() -> None:
    """Raise unless the installed BAF is the pinned version."""
    pinned = pinned_baf_version()
    installed = metadata.version(_BAF_DIST)
    if installed != pinned:
        raise RuntimeError(
            f"{_BAF_DIST} {installed} is installed, but patches/websocket_platform.py is "
            f"vendored from {pinned} (the pin in requirements.txt): refusing to start. "
            "Run: pip install -r requirements.txt"
        )


def install() -> None:
    """Load the vendored WebSocketPlatform as BAF's. Run before ``baf.core.agent``."""
    check_baf_version()
    if "baf.core.agent" in sys.modules:
        raise RuntimeError("baf_patch.install() must run before baf.core.agent is imported")
    current = sys.modules.get(_MODULE)
    if current is not None and _is_patched(getattr(current, "WebSocketPlatform", None)):
        return

    parent = importlib.import_module("baf.platforms.websocket")
    spec = importlib.util.spec_from_file_location(_MODULE, _PATCH_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Vendored BAF patch not found at {_PATCH_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE, None)
        raise
    parent.websocket_platform = module


def verify() -> None:
    """Raise unless the agent will use the patched WebSocketPlatform."""
    from baf.core import agent as agent_module

    if not _is_patched(getattr(agent_module, "WebSocketPlatform", None)):
        raise RuntimeError(
            "BAF WebSocketPlatform is unpatched (no _flush_outbox): refusing to start. "
            "Call app.baf_patch.install() before importing baf.core.agent."
        )
