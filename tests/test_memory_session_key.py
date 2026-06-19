"""Tests for ``memory.memory_session_key`` (B-5).

Conversation memory must be keyed on the v2 payload's stable ``sessionId``
— the BAF session id changes on every WebSocket reconnect, which used to
wipe all multi-turn context. Preference order under test:

  1. ``request.session_id`` from an already-parsed AssistantRequest
  2. the ``sessionId`` parsed from the session's event payload
  3. the BAF ``session.id``
  4. ``id(session)`` as the last resort
"""

import json
import sys
import types

import pytest

from memory import memory_session_key

from tests.conftest import FakeSession


def _ensure_baf_stub_for_adapters():
    """protocol.adapters imports baf at module level; stub the pieces it
    needs so the parse path is exercisable without the BAF stack.

    Installed at test RUN time (not collection) so the pre-existing
    collection errors of baf-dependent test modules are unaffected.
    """
    # Evict MagicMock impostors left in sys.modules by test_confirmation's
    # collection-time stubbing — they would shadow the real protocol package.
    for _name in ("protocol", "protocol.adapters", "protocol.types"):
        _mod = sys.modules.get(_name)
        if _mod is not None and not isinstance(_mod, types.ModuleType):
            sys.modules.pop(_name, None)
    if "baf.core.session" not in sys.modules:
        baf = sys.modules.get("baf") or types.ModuleType("baf")
        baf_core = types.ModuleType("baf.core")
        baf_session = types.ModuleType("baf.core.session")

        class _StubSession:  # pragma: no cover - placeholder type only
            pass

        baf_session.Session = _StubSession
        baf_core.session = baf_session
        baf.core = baf_core
        sys.modules.setdefault("baf", baf)
        sys.modules["baf.core"] = baf_core
        sys.modules["baf.core.session"] = baf_session

    if "baf.library.transition.events.base_events" not in sys.modules:
        baf = sys.modules["baf"]
        baf_library = types.ModuleType("baf.library")
        baf_transition = types.ModuleType("baf.library.transition")
        baf_events = types.ModuleType("baf.library.transition.events")
        base_events = types.ModuleType("baf.library.transition.events.base_events")

        class _StubReceiveJSONEvent:  # pragma: no cover - placeholder type only
            pass

        base_events.ReceiveJSONEvent = _StubReceiveJSONEvent
        baf_events.base_events = base_events
        baf_transition.events = baf_events
        baf_library.transition = baf_transition
        baf.library = baf_library
        sys.modules["baf.library"] = baf_library
        sys.modules["baf.library.transition"] = baf_transition
        sys.modules["baf.library.transition.events"] = baf_events
        sys.modules["baf.library.transition.events.base_events"] = base_events


@pytest.fixture(autouse=True)
def _clean_modules():
    """Every test in this file runs with the real protocol package and a
    baf stub sufficient for ``protocol.adapters``."""
    _ensure_baf_stub_for_adapters()
    yield


def _session_with_payload_session_id(session_id: str) -> FakeSession:
    """FakeSession whose event carries a v2 payload with a sessionId."""
    inner = {
        "action": "user_message",
        "protocolVersion": "2.0",
        "clientMode": "workspace",
        "sessionId": session_id,
        "message": "hello",
        "context": {"activeDiagramType": "ClassDiagram"},
    }
    return FakeSession({
        "action": "user_message",
        "user_id": "baf-user",
        "message": json.dumps(inner),
    })


class _Request:
    def __init__(self, session_id=None):
        self.session_id = session_id


def test_prefers_explicit_request_session_id():
    session = FakeSession()
    session.id = "baf-session-1"
    assert memory_session_key(session, _Request("stable-abc")) == "stable-abc"


def test_parses_payload_session_id_when_no_request_given():
    _ensure_baf_stub_for_adapters()
    session = _session_with_payload_session_id("payload-sid-42")
    assert memory_session_key(session) == "payload-sid-42"


def test_request_without_session_id_falls_back_to_payload_then_baf_id():
    _ensure_baf_stub_for_adapters()
    session = _session_with_payload_session_id("payload-sid-43")
    # request given but has no session_id → payload sessionId wins
    assert memory_session_key(session, _Request(None)) == "payload-sid-43"


def test_falls_back_to_baf_session_id_when_parse_fails():
    session = object.__new__(FakeSession)  # no event/_store: parse raises
    session.id = "baf-session-2"
    assert memory_session_key(session) == "baf-session-2"


def test_last_resort_is_object_identity():
    session = object.__new__(FakeSession)
    key = memory_session_key(session)
    assert key == str(id(session))
