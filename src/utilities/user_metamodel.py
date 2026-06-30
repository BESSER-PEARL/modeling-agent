"""User-profile metamodel loader.

The BESSER editor's *User Profile* perspective (diagram type ``UserDiagram``)
draws its instance boxes from a fixed metamodel that the editor does **not**
transmit to the modeling agent.  We therefore bundle a copy of that metamodel
and load it here as the reference catalog for the user-profile handler —
analogous to how the object-diagram handler uses a reference class diagram.

Source of truth: this file is a verbatim copy of
``packages/editor/src/main/packages/user-modeling/usermetamodel_buml_short.json``
in the BESSER-Web-Modeling-Editor repository.  Keep them in sync — the
``classId`` / ``attributeId`` values the agent emits must match the ids the
editor uses, or the editor cannot link generated attribute rows back to the
schema.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_METAMODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "resources", "user_metamodel.json"
)

# Cached after first successful load (the file never changes at runtime).
_CACHED_METAMODEL: Optional[Dict[str, Any]] = None


def load_user_metamodel() -> Dict[str, Any]:
    """Return the bundled user-profile metamodel as a ClassDiagram-shaped dict.

    The returned object has the same ``{elements, relationships, ...}`` shape
    as any class diagram, so the object-diagram reference-catalog helpers work
    on it unchanged.  Returns an empty ``{"elements": {}}`` dict if the bundled
    resource is missing or unreadable (the handler degrades gracefully).
    """
    global _CACHED_METAMODEL
    if _CACHED_METAMODEL is not None:
        return _CACHED_METAMODEL

    try:
        with open(_METAMODEL_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not isinstance(data.get("elements"), dict):
            logger.error(
                "[UserMetamodel] Bundled metamodel has unexpected shape; "
                "using empty catalog."
            )
            data = {"elements": {}, "relationships": {}}
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(f"[UserMetamodel] Failed to load bundled metamodel: {exc}")
        data = {"elements": {}, "relationships": {}}

    _CACHED_METAMODEL = data
    return _CACHED_METAMODEL
