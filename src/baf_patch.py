"""Install the vendored BAF WebSocket platform (``patches/websocket_platform.py``).

The Docker image copies the patched file over the pip-installed one at build
time. A run outside Docker (local dev, on-prem) would otherwise use stock BAF:
the BYOK key is logged in plain text and ignored, and there is no reply outbox
or ownership-guarded slot reclaim. ``install()`` loads the vendored module under
BAF's own module name before ``baf.core.agent`` imports it, so both paths run
the same code; ``verify()`` refuses to start if that did not take.
"""

import importlib
import importlib.util
import os
import sys

_MODULE = "baf.platforms.websocket.websocket_platform"
_PATCH_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "patches", "websocket_platform.py",
)


def _is_patched(platform_cls) -> bool:
    return platform_cls is not None and hasattr(platform_cls, "_flush_outbox")


def install() -> None:
    """Load the vendored WebSocketPlatform as BAF's. Run before ``baf.core.agent``."""
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
            "Call baf_patch.install() before importing baf.core.agent."
        )
