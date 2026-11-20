"""Bare answers a user types (or a quick-action button sends) to an open question."""
import re

# A message that normalizes to one of these answers a pending confirmation and
# is never a pivot, whatever intent label the classifier stamped on it.
FLOW_ANSWER_TOKENS = {
    "replace", "replace it", "keep", "keep it", "keep both",
    "keep it and add alongside", "add alongside", "new tab",
    "create in a new tab", "confirm", "confirm apply the change",
    "apply the change", "go ahead", "yes", "no", "cancel", "ok",
}


def normalize_flow_token(message: str) -> str:
    """Lowercase, strip punctuation/dashes, collapse whitespace."""
    cleaned = re.sub(r"[^\w\s]", " ", (message or "").lower())
    return " ".join(cleaned.split())


def is_bare_flow_answer(message: str) -> bool:
    return normalize_flow_token(message) in FLOW_ANSWER_TOKENS
