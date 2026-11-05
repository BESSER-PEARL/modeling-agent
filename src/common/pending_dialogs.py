"""Pending dialogs: the questions the assistant is waiting on, and their stashes.

When a turn ends on a question (replace or keep? which GUI mode? run the
Spec-Driven Agent? which generator settings?), the next message may answer it.
Everything that dialog needs lives in ONE BAF session variable,
``common.session_keys.PENDING_DIALOGS``: a dict of the slots below. Read and
write it only through :func:`get`, :func:`put` and :func:`clear`, never with
``session.get`` / ``session.set`` on a slot name.

Several dialogs can be open at once (the mismatch "Update model + generate"
chain keeps the Spec-Driven stash while the rebuild asks replace-or-keep; a
"design X and generate Y" plan keeps its paused generator while the model step
asks the same). :data:`PRECEDENCE` decides which one a message is checked
against first: ``app.confirmation.resume_pending`` resumes them in this order
and the classifier is shown the first open one (``_pending_flow_context``).

Like every BAF session variable, the slots live in process memory only (BAF's
monitoring database is not enabled) and are freed by ``Agent.close_session``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, NamedTuple, Optional

from common.session_keys import PENDING_DIALOGS

# ── Slots ─────────────────────────────────────────────────────────────

# Replace / keep / new-tab question for a create on a non-empty diagram, or the
# destructive-modify confirmation (``destructive_modify`` in the dict). Carries
# the stored operation and any ``remaining_operations`` of its plan.
PENDING_COMPLETE_SYSTEM = "pending_complete_system"
# Basic CRUD pages vs AI design, asked before a GUI build.
PENDING_GUI_CHOICE = "pending_gui_choice"

# Web-app pause: set when a "create a web app" plan builds a GUI.
# The plan's auto-generation op is STRIPPED at the source so nothing can auto-run
# on any execution path; this flag then drives the "generate the web app?" prompt
# once the GUI is built. The user triggers generation explicitly afterwards
# (exactly like the class-diagram flow, which has no generation op in its plan).
PENDING_WEBAPP_GENERATE = "_pending_webapp_generate"

# Generator menu (``_awaiting_selection``) or the generator whose settings are
# being collected, and the settings collected so far.
PENDING_GENERATOR_TYPE = "pending_generator_type"
PENDING_GENERATOR_CONFIG = "pending_generator_config"
CONFIG_PROMPT_ATTEMPTS = "_config_prompt_attempts"
# Marker key stored INSIDE the PENDING_GENERATOR_CONFIG dict (not a slot
# itself) when a mixed "design X and generate Y" plan pauses after the
# model step. It flags that the stashed generator is awaiting the user's
# explicit go-ahead to the injection message's "review or continue with
# generating?" question — handle_generation_request only dispatches it on an
# affirmative answer, and strips the marker so it never reaches a
# trigger_generator config payload. Generalizes the web-app pause
# (PENDING_WEBAPP_GENERATE) to every generator type.
PLAN_GENERATION_CONFIRM_FLAG = "_awaiting_generation_confirmation"

# Spec-Driven Agent confirmation gate: every path that would run the
# Spec-Driven Agent (which spends the USER'S OWN API key) stashes
# the smart-gen payload here and asks for explicit confirmation first.
# Also used by the domain-mismatch handoff: on "Update model + generate"
# the create choke point (execution.model_operations) resumes the stash
# after rebuilding the model.
PENDING_SMART_GEN_INSTRUCTIONS = "_pending_smart_gen_instructions"
PENDING_SMART_GEN_PROVIDER = "_pending_smart_gen_provider"
# Unix timestamp set whenever the stash is (re)created. Stashes older
# than the TTL are rejected so an abandoned flow can never hijack a
# later, unrelated request.
PENDING_SMART_GEN_TIMESTAMP = "_pending_smart_gen_timestamp"
PENDING_SMART_GEN_ORIGINAL_REQUEST = "_pending_smart_gen_original_request"
# The run mode to force on the editor; unset leaves the choice to the editor.
PENDING_SMART_GEN_MODE = "_pending_smart_gen_mode"
# Set to the EXACT rebuild prompt ("create a class diagram for X") when the
# domain-mismatch "Update model + generate" quick action is offered. That
# action rebuilds the class diagram via a plain create; the guard and the
# model-build choke point only keep the stash / resume the smart-gen handoff
# when the incoming message equals this prompt — so a DIFFERENT create typed
# after a mismatch abandons normally instead of spuriously resuming. Consumed
# once, at the matching rebuild.
MISMATCH_REGEN_PENDING = "_mismatch_regen_pending"

# ── Dialogs ───────────────────────────────────────────────────────────

GUI_CHOICE = "gui_choice"
SYSTEM_CONFIRMATION = "system_confirmation"
SMART_GEN = "smart_gen"
GENERATOR = "generator"

# First wins when several dialogs are open.
PRECEDENCE = (GUI_CHOICE, SYSTEM_CONFIRMATION, SMART_GEN, GENERATOR)

# A dialog is open while its slot holds a truthy value; that value is its payload.
_DIALOG_SLOT = {
    GUI_CHOICE: PENDING_GUI_CHOICE,
    SYSTEM_CONFIRMATION: PENDING_COMPLETE_SYSTEM,
    SMART_GEN: PENDING_SMART_GEN_INSTRUCTIONS,
    GENERATOR: PENDING_GENERATOR_TYPE,
}


class PendingDialog(NamedTuple):
    kind: str
    payload: Any


# Plan steps run model operations in threads, and each may open a dialog.
_lock = threading.Lock()


def get(session: Any, slot: str, default: Any = None) -> Any:
    """Value of one slot, or *default* when it is not set."""
    value = (session.get(PENDING_DIALOGS) or {}).get(slot)
    return default if value is None else value


def put(session: Any, slot: str, value: Any) -> None:
    """Set one slot; ``None`` clears it."""
    with _lock:
        slots = dict(session.get(PENDING_DIALOGS) or {})
        if value is None:
            slots.pop(slot, None)
        else:
            slots[slot] = value
        session.set(PENDING_DIALOGS, slots)


def clear(session: Any, *slots: str) -> None:
    """Clear the given slots."""
    with _lock:
        current = dict(session.get(PENDING_DIALOGS) or {})
        for slot in slots:
            current.pop(slot, None)
        session.set(PENDING_DIALOGS, current)


def is_open(session: Any, kind: str) -> bool:
    return bool(get(session, _DIALOG_SLOT[kind]))


def current(session: Any) -> Optional[PendingDialog]:
    """The open dialog a message is checked against first, or ``None``."""
    for kind in PRECEDENCE:
        payload = get(session, _DIALOG_SLOT[kind])
        if payload:
            return PendingDialog(kind, payload)
    return None


# ── One question per parallel plan ────────────────────────────────────

class PlanDialog:
    """The dialog a parallel plan opened: its groups run in threads, and a
    second question would overwrite the first's slot or bury it."""

    def __init__(self) -> None:
        self.slot: Optional[str] = None
        self._lock = threading.Lock()

    def claim(self, slot: str) -> bool:
        with self._lock:
            if self.slot is None:
                self.slot = slot
                return True
            return False


_plan_dialog: ContextVar[Optional[PlanDialog]] = ContextVar("plan_dialog", default=None)


@contextmanager
def one_per_plan() -> Iterator[PlanDialog]:
    """Within this block (and the contexts copied from it) only the first
    model operation that asks a question may open its dialog."""
    plan_dialog = PlanDialog()
    token = _plan_dialog.set(plan_dialog)
    try:
        yield plan_dialog
    finally:
        _plan_dialog.reset(token)


def may_open(slot: str) -> bool:
    """Whether a model operation may open a dialog in *slot* now: always,
    except inside :func:`one_per_plan` once another dialog is open."""
    plan_dialog = _plan_dialog.get()
    return plan_dialog is None or plan_dialog.claim(slot)
