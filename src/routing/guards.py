"""Deterministic routing guards.

The unified classifier decides where a message goes. These are the rules that
decide or short-circuit by protocol field or text match, without an LLM call:
pure functions over the request or message (and the verdict, where a guard
depends on it). Two rule sets live elsewhere on purpose: the classifier's own
post-validation (``unified_classifier._post_validate``) and the diagram-type
keywords of layer 2 (``orchestrator.workspace_orchestrator``).

Before the classifier (``unified_classifier``):
  skips_classification   reconnect replays and blank text are never classified
  frontend_event_verdict generator callbacks, routed by their ``action``
  auto_fix_verdict       the editor's ``[auto-fix]`` repair requests

After routing, in the state that received the message:
  reconcile_intent       create vs modify: the verdict over the state default
  modeling_guard         create / modify states, first match in the order below
  quick_response_key     fallback, greetings and modeling-help states
  is_mixed_modeling_and_generation   generation state
  is_past_generation_question        generation handler
  generation_route       the dispatch route of a verdict in the generation handler
"""

import re
from typing import Any, Dict, Optional, Tuple

from routing.generation_signals import detect_generator_type
from protocol.actions import Action

# ---------------------------------------------------------------------------
# Before the classifier
# ---------------------------------------------------------------------------

# Stamped by the editor's post-injection validation loop on the one repair
# request it sends after a model fails backend validation. Not prose, so it
# never reaches the LLM and the repair loop works during an outage.
AUTO_FIX_PREFIX = "[auto-fix]"


def skips_classification(request: Any) -> bool:
    """True for messages the classifier never sees: the reconnect-recovery
    control message, and blank text that is not a frontend_event (callbacks
    may carry an empty message but still route to the generation handler)."""
    action = getattr(request, "action", None)
    if action == Action.REPLAY_LAST_RESPONSE:
        return True
    return action != Action.FRONTEND_EVENT and not (request.message or "").strip()


def frontend_event_verdict(request: Any) -> Optional[Dict[str, Any]]:
    """Verdict fields for a ``frontend_event`` callback (``generator_result``
    etc.). These are protocol events, not prose: classifying their text sent a
    generation-completion echo to greetings."""
    if getattr(request, "action", None) != Action.FRONTEND_EVENT:
        return None
    return {
        "intent": "generation_intent",
        "generation_route": "other",
        "reason": "frontend_event callback — routed deterministically, no LLM call",
    }


def auto_fix_verdict(message: str) -> Optional[Dict[str, Any]]:
    """Verdict fields for an editor validation repair (``[auto-fix]`` marker):
    always a modify, and never an answer to a pending question."""
    if not message.startswith(AUTO_FIX_PREFIX):
        return None
    return {
        "intent": "modify_model_intent",
        "model_disposition": "extend_existing",
        "pending_flow_action": "new_request",
        "reason": "Editor validation repair ([auto-fix] marker) — deterministic modify route",
    }


# ---------------------------------------------------------------------------
# Create / modify states
# ---------------------------------------------------------------------------

_INTENT_TO_MODE = {
    "modify_model_intent": ("modify_model_intent", "modify_model"),
    "create_complete_system_intent": ("create_complete_system_intent", "complete_system"),
}


def reconcile_intent(intent_name: str, default_mode: str, unified_intent) -> Tuple[str, str]:
    """Return the (intent, mode) to execute with. Honors the unified classifier's
    create/modify verdict over the state default — they differ only when a pending
    flow suppressed intent routing and the message landed in the wrong state."""
    if unified_intent in _INTENT_TO_MODE and unified_intent != intent_name:
        return _INTENT_TO_MODE[unified_intent]
    return intent_name, default_mode


# Unambiguous prompt-subversion / non-modeling phrases. Each has ZERO legitimate
# use in a "model my X" request, so the agent declines in prose instead of
# modeling the injection text (a "RefusalNotice" class, a "Pirate" class).
# Security is enforced elsewhere (instructions are never executed, the prompt
# never leaked). High-precision: real work ("delete the Doctor class") passes.
_NON_MODELING_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|earlier|above)\s+"
    r"(?:instruction|prompt|rule|direction|message)",
    r"disregard\s+(?:your|all|the|any|these|previous|prior)\s+"
    r"(?:rule|instruction|guardrail|previous|prior|direction)",
    r"(?:reveal|show|print|expose|repeat|leak)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+)?(?:prompt|instruction|rule)",
    r"\byou\s+are\s+now\s+(?:a|an)\b",
    r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b",
    r"(?:cat|less|head|tail|nano|vim)\s+/etc/\w+",   # shell exfil attempts
    r"\brm\s+-rf\b",
    r"/etc/passwd",
))


def is_non_modeling(message: str) -> bool:
    """True when the message is a prompt-injection / rule-subversion attempt with
    no legitimate modeling intent (persona hijack, "reveal your prompt", shell
    commands). Such input is declined in prose, not modeled."""
    text = message or ""
    return any(p.search(text) for p in _NON_MODELING_PATTERNS)


# Classifier-OUTAGE net: fires ONLY when the unified classifier reached no
# verdict (provider down / parse failure -> fallback_intent), where the modeling
# states fall back to themselves and would otherwise run a full create on
# "nothing" or "draw me a cat". With a real verdict it never runs.
OUTAGE_DECLINE_PHRASES = {
    "nothing", "no", "nope", "nah", "no thanks", "no thank you",
    "never mind", "nevermind", "nvm", "not now", "stop", "cancel",
    "that's all", "thats all", "i'm good", "im good",
}
OUTAGE_ARTIFACT_RE = re.compile(
    r"\b(?:picture|image|photo|drawing|painting|logo|poem|story|song|joke)s?\s+"
    r"(?:of|about)\b|\btell\s+me\s+a\s+(?:joke|story)\b",
    re.IGNORECASE,
)


# A create request that negates the very content it asks to create. Only
# STRUCTURAL self-negation (classes/elements/nodes/entities/attributes), so
# "a shop with no online payments" is never caught.
_SELF_CONTRADICTORY_CREATE_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:with|having|contain(?:s|ing)?)\s+(?:absolutely\s+|exactly\s+)?"
    r"(?:no|zero|0)\s+(?:class(?:es)?|element(?:s)?|node(?:s)?|entit(?:y|ies)|attribute(?:s)?)\b",
    r"\bwithout\s+(?:any\s+)?(?:class(?:es)?|element(?:s)?|node(?:s)?|entit(?:y|ies)|attribute(?:s)?)\b",
    # "don't model anything" / "model nothing at all" — verb-anchored and
    # requiring "anything"/"nothing at all" so "add nothing fancy" is NOT caught.
    r"\b(?:don'?t|do\s+not|never)\s+(?:model|add|include|create|design)\s+"
    r"(?:anything|any\s+(?:class|element|node|entit))",
    r"\bmodel\s+nothing\b(?!\s+(?:fancy|special|extra|too|but))",
    r"\bempty\s+(?:class\s+|object\s+|state\s+(?:machine\s+)?|agent\s+)?diagram\b",
))


def is_self_contradictory_create(message: str) -> bool:
    """True when a CREATE request negates the content it asks to create
    ("a class diagram with no classes", "model nothing", "an empty diagram")."""
    text = message or ""
    return any(p.search(text) for p in _SELF_CONTRADICTORY_CREATE_PATTERNS)


# Filler words stripped when checking whether a "create …" request names a
# domain. Conservative: any real domain noun ("hospital", "task") survives.
_CREATE_FILLER = frozenset({
    "create", "make", "build", "design", "generate", "add", "new", "start",
    "a", "an", "the", "some", "any", "please", "for", "me", "us", "my", "our",
    "app", "application", "web", "webapp", "site", "website", "system", "software",
    "program", "project", "thing", "something", "stuff", "model", "diagram",
    "can", "you", "could", "would", "will", "i", "we", "want", "need", "like",
    "to", "of", "with", "and", "it", "this", "that", "help",
})


def is_too_vague_create(message: str) -> bool:
    """True when a create request has NO domain content after stripping filler."""
    words = re.findall(r"[a-zA-Z]+", (message or "").lower())
    core = [w for w in words if w not in _CREATE_FILLER]
    # Nothing substantive to model (e.g. "create", "make an app", "build a system").
    return sum(len(w) for w in core) < 3


def modeling_guard(message: str, intent_name: str, verdict: Any) -> Optional[str]:
    """The first guard that stops a create / modify state from modeling the
    message, or None to model it. ``intent_name`` is the reconciled intent and
    ``verdict`` the cached unified classification (None when it never ran).

    Order: ``non_modeling`` (both states), then the outage net
    (``outage_decline``, ``outage_artifact``; only without a usable verdict),
    then ``self_contradictory`` and ``too_vague`` (create only)."""
    if is_non_modeling(message):
        return "non_modeling"
    if verdict is None or getattr(verdict, "intent", None) == "fallback_intent":
        normalized = " ".join(re.sub(r"[^\w\s']", " ", message.lower()).split())
        if normalized in OUTAGE_DECLINE_PHRASES:
            return "outage_decline"
        if OUTAGE_ARTIFACT_RE.search(message):
            return "outage_artifact"
    if intent_name == "create_complete_system_intent":
        if is_self_contradictory_create(message):
            return "self_contradictory"
        if is_too_vague_create(message):
            return "too_vague"
    return None


# ---------------------------------------------------------------------------
# Fallback, greetings and modeling-help states
# ---------------------------------------------------------------------------

# Substrings that pick a canned reply (no LLM call), checked in order.
QUICK_PATTERNS = [
    # What is BESSER?
    (["what is besser", "what's besser", "tell me about besser", "explain besser", "about besser"],
     "what_is_besser"),
    # What can you do?
    (["what can you do", "what do you do", "your capabilities", "what are your features",
      "list your features", "show me what you can do", "what are you capable of",
      "how can you help", "what do you support", "what diagrams"],
     "what_can_you_do"),
    # Help
    (["help me", "i need help", "how does this work", "how do i use",
      "getting started", "quick start", "tutorial", "guide me"],
     "help"),
    # How do I run the generated/downloaded app?
    (["how do i run", "how to run", "how do i start", "how do i launch",
      "run the app", "run the zip", "run the generated", "run the downloaded",
      "run it on my", "run this on my", "start the backend", "start the generated",
      "downloaded the zip", "got the zip", "run the code"],
     "how_to_run"),
]


def quick_response_key(message: str) -> Optional[str]:
    """The canned-reply key the message asks for, or None."""
    msg_lower = message.lower().strip()
    # Exact short matches
    if msg_lower in ("help", "?", "help!", "help?"):
        return "help"
    for patterns, key in QUICK_PATTERNS:
        if any(p in msg_lower for p in patterns):
            return key
    return None


# ---------------------------------------------------------------------------
# Generation state
# ---------------------------------------------------------------------------

def is_mixed_modeling_and_generation(message: str) -> bool:
    """True for "create a class diagram and generate Django": the model is built
    first and generation waits for the user's go-ahead."""
    lower = (message or "").lower()
    if not detect_generator_type(lower):
        return False

    modeling_keywords = [
        "class diagram",
        "object diagram",
        "state machine",
        "state diagram",
        "agent diagram",
        "gui diagram",
        "quantum circuit",
        "create class",
        "create an agent",
        "create state",
        "create model",
        "structural model",
        "model a",
        "design a",
    ]
    has_modeling_language = any(token in lower for token in modeling_keywords)
    has_multi_step_connector = any(token in lower for token in [" and ", " then ", " also ", " after ", " after that ", ";"])
    return has_modeling_language and has_multi_step_connector


# A question ABOUT a past generation: interrogative opener + a
# generated/built/created reference within the same short message, or the
# bare "what did/have/was ... generate(d)" forms. Precision-first — an
# imperative like "generate rust classes" never matches (no interrogative
# opener), and future-directed questions are excluded separately.
_PAST_GEN_QUESTION_RE = re.compile(
    r"^(what|which|show|tell|list|describe|explain)\b.{0,60}?\b(generat|built|created|produced)",
    re.IGNORECASE,
)
# Future-directed words that turn "what ... generate" into a request for
# options/capabilities or a NEW run — those keep the normal routing.
_FUTURE_GEN_WORDS_RE = re.compile(
    r"\b(should|shall|can|could|would|will|next|now|again|regenerate|new)\b",
    re.IGNORECASE,
)
# What follows the finished run in "show what we built and add a logout button".
_SECOND_CLAUSE_RE = re.compile(
    r"(?P<break>[,;.!?]\s+|\s+(?:and|then|but)\s+)(?:(?:and|then|but|also|please)\s+)*(?P<clause>\S.*)",
    re.IGNORECASE | re.DOTALL,
)
# A second clause opening like this only asks more about the run, or is courtesy.
_ASKS_MORE_RE = re.compile(
    r"(?:what|which|how|why|where|when|who|show|tell|list|describe|explain|please|thanks?)\b",
    re.IGNORECASE,
)


def _second_clause_asks_for_work(message: str, question: re.Match) -> bool:
    """True when the message goes on to request something after naming the run.

    Judged by sentence shape, so no list of verbs goes stale. "what did you
    generate and fix?" shares one question; "... and add a logout button" or a
    new sentence does not.
    """
    second = _SECOND_CLAUSE_RE.search(message, question.end())
    clause = second["clause"].rstrip(" .!?") if second else ""
    if not clause or _ASKS_MORE_RE.match(clause):
        return False
    if question[1].lower() in ("what", "which") and second["break"].strip() not in (".", "!", "?"):
        return not message.rstrip().endswith("?") and len(clause.split()) > 1
    return True


def is_past_generation_question(message: str) -> bool:
    """True for a question about a finished run ("what did you generate?"),
    which is answered from its outcome instead of arming a new run."""
    question = _PAST_GEN_QUESTION_RE.search(message)
    return bool(
        question
        and not _FUTURE_GEN_WORDS_RE.search(message)
        and not _second_clause_asks_for_work(message, question)
    )


def generation_route(verdict: Any) -> str:
    """The generation handler's dispatch route for a unified verdict:
    ``smart`` / ``deterministic`` / ``modeling`` / ``other``.

    ``generation_route`` is only meaningful on generation verdicts; the LLM
    often fills it as 'other' on other intents too, which once sent a mismatch
    rebuild ("create a class diagram for a hotel…") to the clarify reply.
    Non-generation verdicts can still reach the handler (pending flows):
    modeling intents run the modeling pipeline, an ERROR fallback (LLM down,
    tagged by ``_safe_fallback``) shows the generator menu, anything else gets
    the clarify reply."""
    route = verdict.generation_route if verdict.intent == "generation_intent" else None
    if route:
        return route
    if verdict.intent in ("create_complete_system_intent", "modify_model_intent"):
        return "modeling"
    if verdict.intent == "generation_intent":
        return "deterministic"
    if verdict.intent == "fallback_intent" and (verdict.reason or "").startswith("[classifier-error]"):
        return "deterministic"
    return "other"
