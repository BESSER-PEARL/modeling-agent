"""Pending dialogs live in one session variable and resume in one order.

Several dialogs can be open at once: the domain-mismatch "Update model +
generate" chain keeps its Spec-Driven stash while the rebuild asks
replace-or-keep, and a "design X and generate Y" plan keeps its paused
generator while the model step asks the same. ``PRECEDENCE`` decides which one
a message is checked against first, for the resume handlers and for the
question the classifier is shown alike.
"""

import pathlib
import re
import threading
import time
from unittest.mock import patch

from app import confirmation
from common import pending_dialogs
from common.pending_dialogs import (
    PENDING_COMPLETE_SYSTEM,
    PENDING_GENERATOR_CONFIG,
    PENDING_GENERATOR_TYPE,
    PENDING_GUI_CHOICE,
    PENDING_SMART_GEN_INSTRUCTIONS,
    PLAN_GENERATION_CONFIRM_FLAG,
)
from common.session_keys import PENDING_DIALOGS
from routing.unified_classifier import _pending_flow_context

from tests.conftest import FakeSession

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _open_all(session):
    pending_dialogs.put(session, PENDING_GUI_CHOICE, {"operation_request": "screens"})
    pending_dialogs.put(session, PENDING_COMPLETE_SYSTEM, {"can_add_tab": True})
    pending_dialogs.put(session, PENDING_SMART_GEN_INSTRUCTIONS, "build x")
    pending_dialogs.put(session, PENDING_GENERATOR_TYPE, "django")
    pending_dialogs.put(session, PENDING_GENERATOR_CONFIG, {PLAN_GENERATION_CONFIRM_FLAG: True})


class TestStore:
    def test_every_slot_lives_under_one_session_key(self):
        session = FakeSession()
        _open_all(session)
        assert list(session._store) == [PENDING_DIALOGS]
        assert session.get(PENDING_GUI_CHOICE) is None  # a slot is not a session key

    def test_put_none_and_clear_remove_a_slot(self):
        session = FakeSession()
        pending_dialogs.put(session, PENDING_GENERATOR_TYPE, "sql")
        pending_dialogs.put(session, PENDING_GENERATOR_CONFIG, {})
        pending_dialogs.put(session, PENDING_GENERATOR_TYPE, None)
        assert pending_dialogs.get(session, PENDING_GENERATOR_TYPE) is None
        assert pending_dialogs.get(session, PENDING_GENERATOR_CONFIG) == {}
        pending_dialogs.clear(session, PENDING_GENERATOR_CONFIG, PENDING_GUI_CHOICE)
        assert session.get(PENDING_DIALOGS) == {}

    def test_falsy_values_are_kept(self):
        session = FakeSession()
        pending_dialogs.put(session, pending_dialogs.CONFIG_PROMPT_ATTEMPTS, 0)
        assert pending_dialogs.get(session, pending_dialogs.CONFIG_PROMPT_ATTEMPTS) == 0

    def test_concurrent_puts_from_plan_threads_all_land(self):
        class SlowSetSession(FakeSession):
            # Widens the read-modify-write window so unlocked puts overwrite each other.
            def set(self, key, value):
                time.sleep(0.001)
                super().set(key, value)

        session = SlowSetSession()
        slots = [f"slot{i}" for i in range(50)]
        threads = [threading.Thread(target=pending_dialogs.put, args=(session, s, True)) for s in slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(pending_dialogs.get(session, s) for s in slots)


class TestPrecedence:
    def test_order(self):
        assert pending_dialogs.PRECEDENCE == (
            pending_dialogs.GUI_CHOICE,
            pending_dialogs.SYSTEM_CONFIRMATION,
            pending_dialogs.SMART_GEN,
            pending_dialogs.GENERATOR,
        )

    def test_current_is_the_first_open_dialog(self):
        session = FakeSession()
        _open_all(session)
        expected = [
            (pending_dialogs.GUI_CHOICE, PENDING_GUI_CHOICE, "gui_choice"),
            (pending_dialogs.SYSTEM_CONFIRMATION, PENDING_COMPLETE_SYSTEM, "replace_confirm"),
            (pending_dialogs.SMART_GEN, PENDING_SMART_GEN_INSTRUCTIONS, "smart_confirm"),
            (pending_dialogs.GENERATOR, PENDING_GENERATOR_TYPE, "generate_confirm"),
        ]
        for kind, slot, question in expected:
            assert pending_dialogs.current(session).kind == kind
            # the classifier is asked about the same dialog the handlers resume
            assert _pending_flow_context(session)["kind"] == question
            pending_dialogs.put(session, slot, None)
        assert pending_dialogs.current(session) is None
        assert _pending_flow_context(session) is None

    def test_classifier_is_not_shown_the_generator_question_for_another_kind(self):
        session = FakeSession()
        pending_dialogs.put(session, "pending_new_kind", {"x": 1})
        with patch.object(pending_dialogs, "PRECEDENCE", pending_dialogs.PRECEDENCE + ("new_kind",)), \
                patch.dict(pending_dialogs._DIALOG_SLOT, {"new_kind": "pending_new_kind"}):
            assert pending_dialogs.current(session).kind == "new_kind"
            assert _pending_flow_context(session) is None

    def test_an_empty_smart_gen_stash_is_not_open(self):
        session = FakeSession()
        pending_dialogs.put(session, PENDING_SMART_GEN_INSTRUCTIONS, "")
        assert not pending_dialogs.is_open(session, pending_dialogs.SMART_GEN)
        assert pending_dialogs.current(session) is None

    def test_resume_tries_open_dialogs_in_order_until_one_consumes(self):
        session = FakeSession()
        _open_all(session)
        calls = []

        def handler(kind, consumes):
            def run(s):
                calls.append(kind)
                return consumes
            return run

        handlers = {
            pending_dialogs.GUI_CHOICE: handler("gui", False),
            pending_dialogs.SYSTEM_CONFIRMATION: handler("system", False),
            pending_dialogs.SMART_GEN: handler("smart", True),
            pending_dialogs.GENERATOR: handler("generator", True),
        }
        with patch.dict(confirmation.RESUME_HANDLERS, handlers):
            assert confirmation.resume_pending(session) is True
        assert calls == ["gui", "system", "smart"]

    def test_resume_skips_closed_dialogs(self):
        session = FakeSession()
        pending_dialogs.put(session, PENDING_GENERATOR_TYPE, "django")
        called = []
        handlers = {k: (lambda s, k=k: called.append(k) or False) for k in pending_dialogs.PRECEDENCE}
        with patch.dict(confirmation.RESUME_HANDLERS, handlers):
            assert confirmation.resume_pending(session) is False
        assert called == [pending_dialogs.GENERATOR]


def test_src_never_reads_a_slot_as_a_session_key():
    """A slot read with ``session.get`` is always None: the slot is not a key."""
    names = [n for n in dir(pending_dialogs) if n.isupper() and n not in (
        "PRECEDENCE", "GUI_CHOICE", "SYSTEM_CONFIRMATION", "SMART_GEN", "GENERATOR",
        "PLAN_GENERATION_CONFIRM_FLAG", "PENDING_DIALOGS")]
    pattern = re.compile(r"\.(get|set|delete)\(\s*(?:pending_dialogs\.)?(" + "|".join(names) + r")\b")
    assert pattern.search("session.get(PENDING_GUI_CHOICE)")
    assert pattern.search("session.set(pending_dialogs.PENDING_GUI_CHOICE, x)")
    offenders = [
        f"{path.relative_to(SRC)}:{n}"
        for path in SRC.rglob("*.py") if path.name != "pending_dialogs.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []
