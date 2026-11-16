"""The project's trace digest as prompt text, and the artifact a question is about.

The editor summarizes the project trace into ``context.traceDigest``;
``protocol.adapters._parse_trace_digest`` validates and caps it into
``WorkspaceContext.trace_digest``. This module renders that dict into the
compact record the trace-question prompt quotes, and picks the one focus
artifact the ``show_trace`` button may open. Pure and deterministic: the same
digest always renders the same text, cut the same way.

Every string in the digest is user data (element names, requirement quotes,
prompt summaries). The renderer keeps each on one line and turns every angle
bracket in it into a guillemet, after undoing what could hide one (HTML
entities, fullwidth and other compatibility forms, zero-width and bidi
characters), so a value can neither close the record (or open any other tag)
nor pose as a section of it.
"""

import functools
import html
import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

from common.agent_config import TRACE_DIGEST_PROMPT_MAX_CHARS

# The tag the prompt wraps the record in (prompts.traceability).
RECORD_TAG = "trace_record"
_SPACE_RE = re.compile(r"\s+")
# What a "<" or ">" in a value becomes.
_ANGLE_BRACKETS = str.maketrans({"<": "‹", ">": "›"})
# Undoing entities and compatibility forms can expose another layer
# ("&amp;lt;", "＆lt;"); a few rounds reach the fixed point of any real value,
# and an ampersand still left after them is defused (fullwidth).
_UNWRAP_ROUNDS = 8

# Plain-words meaning of each out-of-date reason, shown next to the reason.
OUT_OF_DATE_MEANING = {
    "SOURCE_CHANGED": "the source changed after the link was made",
    "TARGET_CHANGED": "the target was edited after the link was made",
    "UPSTREAM_CHANGED": "something the source was derived from changed",
}

# Room kept for the "not shown" line of a section the budget cuts.
_OMITTED_NOTE_RESERVE = 100


def _visible(text: str) -> str:
    """*text* without format characters (category Cf: zero-width spaces and
    joiners, bidi controls, the BOM, soft hyphens)."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _clean(value: Any) -> str:
    """One line of user data with no angle bracket in it: HTML entities
    decoded, NFKC-normalized (fullwidth "＜" is "<"), format characters
    removed (to a fixed point), whitespace collapsed, then every "<" / ">"
    made "‹" / "›"."""
    return _clean_text(str(value))


# The budget re-renders a focus entry each time it shortens one of its lists.
@functools.lru_cache(maxsize=4096)
def _clean_text(text: str) -> str:
    for _ in range(_UNWRAP_ROUNDS):
        unwrapped = _visible(html.unescape(unicodedata.normalize("NFKC", text)))
        if unwrapped == text:
            break
        text = unwrapped
    else:
        text = text.replace("&", "＆")
    return _SPACE_RE.sub(" ", text).strip().translate(_ANGLE_BRACKETS)


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


def _cut_line(count: int, what: str) -> str:
    return f"   - ... {count} {what} not shown (the record was cut to fit)"


def _focus_entry(index: int, item: Dict[str, Any], cut: Optional[Dict[str, int]] = None) -> str:
    """One focus artifact; *cut* counts the entries of each of its lists the
    budget removed (the oldest history, the last sources and code)."""
    cut = cut or {}
    kind = _clean(item.get("elementType") or "Element")
    lines = [f"[{index}] {kind} {_quoted(item['name'])} (artifact id {_clean(item['artifactId'])})"]
    history = item.get("history") or []
    if history or cut.get("history"):
        lines.append("  history (oldest first):")
        if cut.get("history"):
            lines.append(_cut_line(cut["history"], "older changes"))
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
    if sources or cut.get("derivedFrom"):
        lines.append("  derived from:")
        for source in sources:
            text = f"   - {_clean(source.get('type') or 'Artifact')} {_quoted(source.get('name', ''))}"
            if source.get("evidence"):
                text += f" (evidence: {_quoted(source['evidence'])})"
            lines.append(text)
        if cut.get("derivedFrom"):
            lines.append(_cut_line(cut["derivedFrom"], "more sources"))
    derived = item.get("derived") or []
    if derived or cut.get("derived"):
        lines.append("  derived from it (generated code):")
        for target in derived:
            text = f"   - {_clean(target['path'])}"
            if target.get("lines"):
                text += f" lines {_clean(target['lines'])}"
            text += (f" OUT OF DATE: {_reasons(target['outOfDate'])}" if target.get("outOfDate")
                     else " up to date")
            lines.append(text)
        if cut.get("derived"):
            lines.append(_cut_line(cut["derived"], "more generated fragments"))
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


# The sections, in the order they are shown.
_FOCUS_TITLE = "ELEMENTS THE QUESTION NAMES (index, then what the trace records about each)"
_SECTIONS = (
    ("focus", None),
    ("outOfDate", _out_of_date_entry),
    ("requirements", _requirement_entry),
    ("generations", _generation_entry),
    ("recentOperations", _operation_entry),
)
# The order the budget shortens them in, the least valuable first (the
# editor's fitDigest, shared/traceability/traceDigest.ts): the question is
# about the focus, so it is cut last.
_CUT_ORDER = ("recentOperations", "generations", "requirements", "outOfDate")
# A focus entry's lists, in the order a tie on length shortens them.
_FOCUS_LISTS = ("derived", "derivedFrom", "history")


def _title(key: str, total: int) -> str:
    return {
        "focus": _FOCUS_TITLE,
        "outOfDate": "OUT-OF-DATE TRACE LINKS (source -> target)",
        "requirements": f"REQUIREMENTS ({total} listed)",
        "generations": "CODE GENERATIONS",
        "recentOperations": "RECENT OPERATIONS (newest first)",
    }[key]


def _block(title: str, entries: List[str], omitted: int) -> str:
    if not entries:
        return f"{title}: {omitted} entr{'y' if omitted == 1 else 'ies'} not shown (the record was cut to fit)"
    block = title + ":\n" + "\n".join(entries)
    if omitted:
        block += f"\n- ... {omitted} more not shown (the record was cut to fit)"
    return block


class _Focus:
    """A focus entry the budget may shorten: its lists, what was cut of each,
    and its rendering."""

    def __init__(self, index: int, item: Dict[str, Any]) -> None:
        self.index = index
        self.item = {**item, **{key: list(item.get(key) or []) for key in _FOCUS_LISTS}}
        self.cut = {key: 0 for key in _FOCUS_LISTS}
        self.text = _focus_entry(index, self.item, self.cut)

    def shorten(self) -> bool:
        """Remove one entry of its longest list (the oldest change of the
        history, the last source or fragment); False when all are empty."""
        key = max(_FOCUS_LISTS, key=lambda k: (len(self.item[k]), -_FOCUS_LISTS.index(k)))
        if not self.item[key]:
            return False
        self.item[key].pop(0 if key == "history" else -1)
        self.cut[key] += 1
        self.text = _focus_entry(self.index, self.item, self.cut)
        return True


def render_trace_digest(digest: Dict[str, Any], max_chars: int = TRACE_DIGEST_PROMPT_MAX_CHARS) -> str:
    """The digest as compact prompt text of at most *max_chars* characters.

    Sections are shown in a fixed order (focus elements, out-of-date links,
    requirements, generations, recent operations), entries in the digest's
    order. Over the budget, lists are shortened from their end, the least
    valuable first, as the editor's own ``fitDigest`` does: recent
    operations, generations, requirements, out-of-date links, and last the
    focus, whose element's own lists (the oldest history first, then its
    last sources and code, the longest list first) are cut before an entry is
    dropped, the last entry first. A cut section keeps one line saying how
    many entries are not shown. The header (counts) is charged to the budget
    too. Deterministic: the same digest is always cut the same way.
    """
    header = _header(digest)[:max_chars]
    totals = {key: len(digest.get(key) or []) for key, _ in _SECTIONS}
    rendered = {
        key: [render(item) for item in digest.get(key) or []]
        for key, render in _SECTIONS if render is not None
    }
    focus = [_Focus(i, item) for i, item in enumerate(digest.get("focus") or [])]

    def text() -> str:
        parts = [header]
        for key, _ in _SECTIONS:
            if not totals[key]:
                continue
            entries = [entry.text for entry in focus] if key == "focus" else rendered[key]
            parts.append("\n" + _block(_title(key, totals[key]), entries, totals[key] - len(entries)))
        return "\n".join(parts)

    out = text()
    for key in _CUT_ORDER:
        while rendered[key] and len(out) > max_chars:
            rendered[key].pop()
            out = text()
    while focus and len(out) > max_chars:
        if not focus[-1].shorten():
            focus.pop()
        out = text()
    # Only the one-line notes are left: a budget smaller than those is cut hard.
    return out[:max_chars]


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
