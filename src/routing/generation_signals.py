"""Generation vocabulary and session signals shared by the router, the
request planner and the generation handler.

Kept below all of them so ``unified_classifier`` and ``orchestrator`` can
read them without importing ``handlers``.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from baf.core.session import Session

from session_keys import LAST_SMART_GEN_AT, LAST_SMART_GEN_PROJECT_ID

logger = logging.getLogger(__name__)


# Generator prerequisites are shared with the request planner. Keeping the
# contract here lets both direct generation and multi-step plans validate the
# same diagram requirements without a circular import.
GENERATOR_PREREQUISITES: Dict[str, List[str]] = {
    "web_app": ["ClassDiagram", "GUINoCodeDiagram"],
    "react": ["ClassDiagram", "GUINoCodeDiagram"],
    "django": ["ClassDiagram"],
    "backend": ["ClassDiagram"],
    "sql": ["ClassDiagram"],
    "sqlalchemy": ["ClassDiagram"],
    "python": ["ClassDiagram"],
    "java": ["ClassDiagram"],
    "pydantic": ["ClassDiagram"],
    "jsonschema": ["ClassDiagram"],
    "smartdata": ["ClassDiagram"],
    "rest_api": ["ClassDiagram"],
    "rdf": ["ClassDiagram"],
    "agent": ["AgentDiagram"],
    "qiskit": ["QuantumCircuitDiagram"],
}


GENERATOR_KEYWORDS: Dict[str, List[str]] = {
    "django": ["django"],
    "web_app": [
        "web app",
        "web application",
        "frontend app",
        "frontend generator",
        "gui app",
        "gui generator",
        "graphical ui",
        "generate ui",
        "generate gui",
        "grapesjs",
    ],
    "backend": ["full backend", "backend"],
    "sqlalchemy": ["sqlalchemy", "sql alchemy"],
    "sql": ["database", "relational database", "db schema", "sql ddl", "sql schema", "generate sql", "sql"],
    "python": ["python classes", "generate python"],
    "java": ["java classes", "generate java"],
    "pydantic": ["pydantic"],
    "jsonschema": ["json schema", "jsonschema"],
    "smartdata": ["smart data", "smartdata"],
    "agent": ["besser agent", "agent generator", "generate agent"],
    "qiskit": ["qiskit", "quantum code", "quantum generator", "quantum circuit code", "ibm quantum"],
    "rest_api": ["rest api", "rest_api", "generate rest api"],
    "rdf": ["rdf", "rdf generator", "rdf vocabulary", "generate rdf"],
    "export": [
        "export project", "export the project", "export my project",
        "export to json", "export into json", "export as json", "export json",
        "export to buml", "export into buml", "export as buml", "export buml",
        "export model", "export the model", "export my model",
        "download project", "download the project", "download my project",
        "save as json", "save as buml", "save project",
        "save the project", "save my project", "save project to json",
        "save the project to json", "save the project as json",
        "export diagram", "export the diagram",
    ],
    "deploy": [
        "deploy to render", "deploy on render", "deploy app", "deploy the app",
        "deploy application", "deploy the application", "deploy my app",
        "deploy to cloud", "deploy project", "deploy the project",
        "deploy my project", "render deploy", "publish app", "publish the app",
        "publish to render", "publish my app",
        "deploy this model", "deploy the model", "deploy my model",
        "deploy it", "deploy this", "go ahead and deploy",
        "push this to prod", "push to prod", "push it to prod",
        "ship it to production", "ship this to production",
    ],
}


# Regex fallback patterns for natural phrasing that keyword lists may miss.
# These are tried only when no exact keyword matches.
_FUZZY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("export", re.compile(
        r"\b(?:export|download|save)\b.*\b(?:json|buml|project|model|diagram)\b", re.I)),
    ("deploy", re.compile(
        r"\b(?:deploy|publish|push|ship)\b.*\b(?:render|cloud|app|application|"
        r"project|model|prod|production|live)\b", re.I)),
]


def detect_generator_type(message: str) -> Optional[str]:
    """Detect a code-generator keyword in *message*.

    This is a **pure detection** function — it returns the first matching
    generator type without judging whether the overall request is really a
    code-generation request.  Higher-level callers (``should_route_to_generation``,
    ``handle_generation_request``) apply contextual guards such as
    ``_is_modeling_request`` and ``_is_diagram_creation_request``.
    """
    lower = (message or "").lower()

    # 1. Exact keyword matching (fast path) — use word-boundary-aware check
    #    for short/ambiguous keywords to avoid substring false positives
    #    (e.g. "sql" matching inside "sqlalchemy").
    _BOUNDARY_KEYWORDS = {"sql", "backend"}
    for generator_type, keywords in GENERATOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in _BOUNDARY_KEYWORDS:
                # Word-boundary match to avoid substring collisions
                if re.search(r'\b' + re.escape(keyword) + r'\b', lower):
                    return generator_type
            else:
                if keyword in lower:
                    return generator_type
    # 2. Regex fallback for flexible phrasing
    for generator_type, pattern in _FUZZY_PATTERNS:
        if pattern.search(lower):
            return generator_type
    return None


def _active_project_id(session: Session) -> Optional[str]:
    """Id of the project currently open in the workspace, or ``None``.

    The frontend sends the whole project object as
    ``context.projectSnapshot``, and ``BesserProject.id`` survives the
    client-side context compaction, so the id is available on every
    ``user_message`` turn. It is NOT available on ``frontend_event``
    callbacks — those carry no ``context`` at all — which is why a run's
    project is recorded when the run is ARMED rather than when it completes.

    Returns ``None`` whenever the id cannot be seen (older frontend, widget
    mode, voice context without a snapshot). Callers must treat ``None`` as
    "no project identity" and fail CLOSED, never as a wildcard match.
    """
    # Deferred import: protocol.adapters is cycle-free from here and the
    # parse is cached on the session event, so this is ~free per turn.
    from protocol.adapters import parse_assistant_request

    try:
        request = parse_assistant_request(session)
    except Exception:  # pragma: no cover — defensive; copy must never crash
        logger.debug("Could not resolve the active project id", exc_info=True)
        return None
    snapshot = getattr(getattr(request, "context", None), "project_snapshot", None)
    candidate = snapshot.get("id") if isinstance(snapshot, dict) else None
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def _smart_gen_project_has_app(session: Session, project_id: Optional[str]) -> bool:
    """True when a smart-gen run SUCCEEDED for the project now open.

    This is the only admissible evidence that there is an app to fix. It fails
    CLOSED: an unknown project id (``None`` on either side) never matches, so
    a workspace the agent cannot identify is treated as "no app yet" rather
    than as a wildcard.

    Note what is deliberately NOT evidence: a model/spec having been designed
    in this project (that produces no code), and a stashed pending smart-gen
    (an armed run is not a finished one).
    """
    recorded = session.get(LAST_SMART_GEN_PROJECT_ID)

    # MIGRATION ALLOWANCE. Runs that finished before project attribution existed
    # recorded only LAST_SMART_GEN_AT, with no project id at all. Failing closed
    # on those is a regression, not safety: "fix my app" would lose the
    # SMART-GEN FOLLOW-UP signal and be answered as a help question instead of
    # arming a fix.
    #
    # Deliberately narrow and self-healing: it applies ONLY when no project was
    # ever recorded, and the very next run writes one, after which strict pairing
    # resumes permanently. The caller's recency window bounds it further. A
    # recorded id that MISMATCHES still fails closed — that is the actual bug
    # this gate exists to prevent.
    if recorded is None and session.get(LAST_SMART_GEN_AT) is not None:
        return True

    if not project_id:
        return False
    return recorded == project_id


def recent_smart_gen_for_project(session: Session, window_seconds: float) -> bool:
    """True when a smart-gen run succeeded for the CURRENTLY OPEN project
    within ``window_seconds``.

    ``LAST_SMART_GEN_AT`` on its own is not usable for this: the BAF session is
    keyed on the stable per-browser user_id and survives a project switch (the
    frontend only rotates the payload sessionId, which scopes conversation
    memory), so the bare timestamp means "something was generated recently
    somewhere", not "this project has an app". Pair it with the project the run
    was attributed to — see ``LAST_SMART_GEN_PROJECT_ID``.
    """
    project_id = _active_project_id(session)
    if not _smart_gen_project_has_app(session, project_id):
        return False
    ts = session.get(LAST_SMART_GEN_AT)
    return (
        isinstance(ts, (int, float))
        and not isinstance(ts, bool)
        and (time.time() - ts) <= window_seconds
    )


# --- Continue-from-GitHub guard ------------------------------------------
# Chat path for resuming a PAST generation that was pushed to GitHub:
# "continue from github.com/owner/repo" must hand the frontend a
# trigger_github_import action (the frontend calls the backend import
# endpoint, loads the returned project, and arms the modify machinery —
# the agent itself never touches GitHub or any HTTP endpoint).
# Deterministic and precision-first, like the past-generation guard in
# handlers/generation_handler.py:
# an LLM verdict is never allowed to invent, miss, or swallow an import.
#
# URL form — unambiguous on its own: github.com/{owner}/{repo} with an
# optional scheme, optional ".git" suffix (stripped in the extractor) and
# optional /tree/{branch} segment. The repo pattern is dotted-segment
# shaped so a trailing sentence period is NOT swallowed
# ("...from github.com/x/y." → repo "y") while dotted repo names and
# ".git" still match. The lookbehind rejects lookalike hosts
# ("mygithub.com", "api.github.com").
_GITHUB_URL_RE = re.compile(
    r"(?<![A-Za-z0-9.-])(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})?)/"
    r"(?P<repo>[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)"
    r"(?:/tree/(?P<branch>[^\s?#]+))?",
    re.IGNORECASE,
)
# Continuation verbs (EN + FR "reprendre" forms) — the bare owner/repo
# form fires ONLY alongside one of these, and the classifier-side reroute
# (unified_classifier._post_validate) requires one even for URLs, so a
# "create a diagram like github.com/x/y" style request is never hijacked.
_GITHUB_CONTINUE_VERB_RE = re.compile(
    r"\b(?:continue|continuing|continues|resume|resuming|resumes|"
    r"load|loading|loads|import|importing|imports|"
    r"open|opening|opens|reprend\w*|reprise)\b",
    re.IGNORECASE,
)
# The word that makes a bare "owner/repo" mean a REPOSITORY. Without it
# (plus a continuation verb) a slash pair in ordinary prose
# ("src/handlers", "1/2") must never fire.
_GITHUB_REPO_WORD_RE = re.compile(
    r"\b(?:repos?|repository|repositories|github)\b", re.IGNORECASE,
)
# Bare {owner}/{repo}: same owner/repo shapes as the URL form; the
# lookarounds keep it from matching inside a longer path or URL.
_BARE_OWNER_REPO_RE = re.compile(
    r"(?<![\w./@-])"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})?)/"
    r"(?P<repo>[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)"
    r"(?![\w/-])",
)
# "branch <name>" / "branche <name>" (FR) named in the message text — used
# only when the URL carried no /tree/<branch> segment.
_GITHUB_BRANCH_WORD_RE = re.compile(
    r"\bbranche?\s+[\"'`]?"
    r"(?P<branch>[A-Za-z0-9_/-]+(?:\.[A-Za-z0-9_/-]+)*)",
    re.IGNORECASE,
)


def _extract_github_reference(
    message: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Extract ``(owner, repo, branch)`` from a GitHub reference in *message*.

    Two accepted shapes, precision-first (never fires on arbitrary "a/b"
    prose):

    * a github.com URL (scheme optional, ``.git`` and ``/tree/{branch}``
      suffixes optional) — unambiguous on its own;
    * a bare ``owner/repo`` — ONLY when the message ALSO contains a
      continuation verb (continue / resume / load / import / open /
      reprend…) AND a repo word (repo / repository / github).

    ``branch`` comes from the URL's ``/tree/<branch>`` segment, else a
    "branch <name>" phrase in the message, else ``None`` (the backend
    then uses the repo's default branch). Returns ``None`` when the
    message carries no GitHub reference.
    """
    msg = message or ""
    owner = repo = branch = None
    url_match = _GITHUB_URL_RE.search(msg)
    if url_match:
        owner = url_match.group("owner")
        repo = url_match.group("repo")
        branch = url_match.group("branch")
    elif _GITHUB_CONTINUE_VERB_RE.search(msg) and _GITHUB_REPO_WORD_RE.search(msg):
        bare_match = _BARE_OWNER_REPO_RE.search(msg)
        if bare_match:
            owner = bare_match.group("owner")
            repo = bare_match.group("repo")
    if not owner or not repo:
        return None
    if repo.lower().endswith(".git"):
        repo = repo[: -len(".git")]
    if not repo:
        return None
    if not re.search(r"[A-Za-z]", owner + repo):
        # Digits-only pairs ("1/2", "3/4") are prose — fractions or dates —
        # not a plausible GitHub reference.
        return None
    if branch:
        # A /tree/<branch> segment is greedy up to whitespace so slashed
        # branch names ("feature/login") survive; trim sentence punctuation.
        branch = branch.rstrip(".,;:!?)'\"/")
    if not branch:
        word_match = _GITHUB_BRANCH_WORD_RE.search(msg)
        if word_match:
            branch = word_match.group("branch")
    return owner, repo, branch or None
