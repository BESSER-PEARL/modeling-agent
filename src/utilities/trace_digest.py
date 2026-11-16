"""The project's trace digest as prompt text, and the artifact a question is about.

The editor summarizes the project trace into ``context.traceDigest``;
``protocol.adapters._parse_trace_digest`` validates and caps it into
``WorkspaceContext.trace_digest``. This module renders that dict into the
compact record the trace-question prompt quotes, and picks the one focus
artifact the ``show_trace`` button may open. Pure and deterministic: the same
digest always renders the same text, cut the same way.

Every string in the digest is user data (element names, requirement quotes,
prompt summaries). The renderer keeps each on one line and neutralizes the
record's own delimiter, so a value can neither close the record nor pose as a
section of it.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from common.agent_config import TRACE_DIGEST_PROMPT_MAX_CHARS

# The tag the prompt wraps the record in (prompts.traceability).
RECORD_TAG = "trace_record"
_TAG_RE = re.compile(r"<\s*/?\s*" + RECORD_TAG, re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

# Plain-words meaning of each out-of-date reason, shown next to the reason.
OUT_OF_DATE_MEANING = {
    "SOURCE_CHANGED": "the source changed after the link was made",
    "TARGET_CHANGED": "the target was edited after the link was made",
    "UPSTREAM_CHANGED": "something the source was derived from changed",
}

# Room kept for the "not shown" line of a section the budget cuts.
_OMITTED_NOTE_RESERVE = 100


def _clean(value: Any) -> str:
    """One line of user data: whitespace collapsed, the record tag defused."""
    text = _SPACE_RE.sub(" ", str(value)).strip()
    return _TAG_RE.sub("‹" + RECORD_TAG, text)


def _quoted(value: Any) -> str:
    """Free text (a requirement, a quote, a summary) as a quoted JSON string."""
    return json.dumps(_clean(value), ensure_ascii=False)


def _names(values: Any) -> str:
    return ", ".join(_clean(v) for v in values or [])


def _reasons(reasons: Any) -> str:
    return "; ".join(f"{r} ({OUT_OF_DATE_MEANING.get(r, '')})" for r in reasons or [])


def _header(digest: Dict[str, Any]) -> str:
    parts = [f"Recording: {'on' if digest.get('recording', True) else 'off'}"]
    counts = digest.get("counts") or {}
    totals = [
        f"{label} {counts[key]}"
        for key, label in (("artifacts", "artifacts"), ("links", "trace links"),
                           ("outOfDateLinks", "out-of-date links"))
        if key in counts
    ]
    if totals:
        parts.append("Totals: " + ", ".join(totals))
    operations = counts.get("operations") or {}
    if operations:
        parts.append("Operations: " + ", ".join(f"{k} {v}" for k, v in operations.items()))
    return "\n".join(parts)


def _focus_entry(index: int, item: Dict[str, Any]) -> str:
    kind = _clean(item.get("elementType") or "Element")
    lines = [f"[{index}] {kind} {_quoted(item['name'])} (artifact id {_clean(item['artifactId'])})"]
    history = item.get("history") or []
    if history:
        lines.append("  history (oldest first):")
        for change in history:
            text = f"   - {change['kind']}"
            if change.get("feature"):
                text += f" {_clean(change['feature'])}"
            if "old" in change or "new" in change:
                text += f": {_quoted(change.get('old', ''))} -> {_quoted(change.get('new', ''))}"
            for key, label in (("at", "at"), ("by", "by"), ("via", "via")):
                if change.get(key):
                    text += f" {label} {_clean(change[key])}"
            lines.append(text)
    sources = item.get("derivedFrom") or []
    if sources:
        lines.append("  derived from:")
        for source in sources:
            text = f"   - {_clean(source.get('type') or 'Artifact')} {_quoted(source.get('name', ''))}"
            if source.get("evidence"):
                text += f" (evidence: {_quoted(source['evidence'])})"
            lines.append(text)
    derived = item.get("derived") or []
    if derived:
        lines.append("  derived from it (generated code):")
        for target in derived:
            text = f"   - {_clean(target['path'])}"
            if target.get("lines"):
                text += f" lines {_clean(target['lines'])}"
            text += (f" OUT OF DATE: {_reasons(target['outOfDate'])}" if target.get("outOfDate")
                     else " up to date")
            lines.append(text)
    if len(lines) == 1:
        lines.append("  (the trace records nothing else about it)")
    return "\n".join(lines)


def _out_of_date_entry(item: Dict[str, Any]) -> str:
    text = f"- {_clean(item.get('source', '?'))} -> {_clean(item.get('target', '?'))}"
    if item.get("reasons"):
        text += f" [{_reasons(item['reasons'])}]"
    if item.get("generation"):
        text += f" (made by {_clean(item['generation'])})"
    return text


def _requirement_entry(item: Dict[str, Any]) -> str:
    text = "- " + _clean(item.get("id", "?"))
    if item.get("kind"):
        text += f" [{_clean(item['kind'])}]"
    if item.get("text"):
        text += f" {_quoted(item['text'])}"
    if item.get("realizedBy"):
        text += f"; realized by: {_names(item['realizedBy'])}"
    if item.get("covered") is False:
        text += "; NOT COVERED by the model"
    elif item.get("covered") is True:
        text += "; covered"
    if item.get("quote"):
        text += f"; the user's words: {_quoted(item['quote'])}"
    return text


def _generation_entry(item: Dict[str, Any]) -> str:
    text = f"- {_clean(item['generator'])}"
    if item.get("at"):
        text += f" at {_clean(item['at'])}"
    if "files" in item:
        text += f": {item['files']} file(s)"
    if "outOfDateFiles" in item:
        text += f", {item['outOfDateFiles']} out of date"
    return text


def _operation_entry(item: Dict[str, Any]) -> str:
    text = f"- {_clean(item.get('type', 'Operation'))}"
    if item.get("status"):
        text += f" ({_clean(item['status'])})"
    if item.get("startedAt"):
        text += f" at {_clean(item['startedAt'])}"
    if item.get("by"):
        text += f" by {_names(item['by'])}"
    if item.get("tools"):
        text += f" with {_names(item['tools'])}"
    if item.get("summary"):
        text += f": {_quoted(item['summary'])}"
    return text


def _sections(digest: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """(title, entries) in the order they are kept when the record is cut."""
    requirements = digest.get("requirements") or []
    return [
        ("ELEMENTS THE QUESTION NAMES (index, then what the trace records about each)",
         [_focus_entry(i, item) for i, item in enumerate(digest.get("focus") or [])]),
        ("OUT-OF-DATE TRACE LINKS (source -> target)",
         [_out_of_date_entry(item) for item in digest.get("outOfDate") or []]),
        (f"REQUIREMENTS ({len(requirements)} listed)",
         [_requirement_entry(item) for item in requirements]),
        ("CODE GENERATIONS", [_generation_entry(item) for item in digest.get("generations") or []]),
        ("RECENT OPERATIONS (newest first)",
         [_operation_entry(item) for item in digest.get("recentOperations") or []]),
    ]


def render_trace_digest(digest: Dict[str, Any], max_chars: int = TRACE_DIGEST_PROMPT_MAX_CHARS) -> str:
    """The digest as compact prompt text of at most about *max_chars* characters.

    Sections come in a fixed priority order (focus elements, out-of-date
    links, requirements, generations, recent operations) and entries in the
    digest's order. When the budget runs out, the rest of a section is
    replaced by one line saying how many entries are not shown, and the next
    sections are filled from what is left, so the cut is deterministic.
    """
    out = [_header(digest)]
    used = len(out[0])
    sections = [(title, entries) for title, entries in _sections(digest) if entries]
    for position, (title, entries) in enumerate(sections):
        # Room for the one-line note of every later section, so each is at least counted.
        available = max_chars - _OMITTED_NOTE_RESERVE * (len(sections) - position - 1)
        kept: List[str] = []
        cost = used + len(title) + 2
        for entry in entries:
            if cost + len(entry) + 1 + _OMITTED_NOTE_RESERVE > available:
                break
            kept.append(entry)
            cost += len(entry) + 1
        omitted = len(entries) - len(kept)
        if not kept:
            note = f"\n{title}: {omitted} entr{'y' if omitted == 1 else 'ies'} not shown (the record was cut to fit)"
            if used + len(note) <= max_chars:
                out.append(note)
                used += len(note)
            continue
        block = "\n" + title + ":\n" + "\n".join(kept)
        if omitted:
            block += f"\n- ... {omitted} more not shown (the record was cut to fit)"
        out.append(block)
        used += len(block)
    return "\n".join(out)


def _mentions(message: str, name: str) -> bool:
    """True when *name* appears in *message* as a whole name (case-insensitive)."""
    pattern = r"(?<![\w.])" + re.escape(name) + r"(?!\w)"
    return re.search(pattern, message, re.IGNORECASE) is not None


def pick_focus_artifact(digest: Optional[Dict[str, Any]], message: str) -> Optional[Dict[str, Any]]:
    """The one focus artifact the question is about, else None.

    The editor puts in ``focus`` the artifacts whose name appears in the
    message, so a single entry is the answer. With several, the one entry the
    message names wins, a member over its owner ("Book.isbn" over "Book");
    two or more named entries (or two with the same name) are no answer.
    Only ever an entry of the digest, so an artifact id never comes from the
    model.
    """
    focus = (digest or {}).get("focus") or []
    if len(focus) == 1:
        return focus[0]
    named = [item for item in focus if _mentions(message or "", item["name"])]
    # "Book" named only as part of "Book.isbn" is not a second artifact.
    named = [
        item for item in named
        if not any(other["name"].startswith(item["name"] + ".") for other in named)
    ]
    return named[0] if len(named) == 1 else None
