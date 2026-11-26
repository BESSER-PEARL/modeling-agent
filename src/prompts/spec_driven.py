"""Spec-Driven Agent hand-off: the instructions sent with ``trigger_smart_generator``.

The two headings are a contract with the harness (BESSER
``besser/spec_driven_agent/planning/requirements_ledger.py``): it takes the
user's own text from after ``VERBATIM_MARKER`` and the user's newest message
from after the last ``LATER_REQUEST_MARKER``.
"""

VERBATIM_MARKER = "## The user's original request, verbatim"
LATER_REQUEST_MARKER = "## Subsequent user request, verbatim"


def with_later_request(previous: str, message: str) -> str:
    """``previous`` followed by the user's later ``message``."""
    return (
        f"{previous}\n\n{LATER_REQUEST_MARKER}\n\n"
        "This later request takes precedence where it changes earlier requirements.\n\n"
        f"{message}"
    )


def framed_instructions(instructions: str, original: str, starting_point: str = "") -> str:
    """The classifier's summary, labelled as possibly inaccurate, then the
    run's starting point, then the user's verbatim request as the authority."""
    return (
        "## Assistant's notes (may be inaccurate)\n\n"
        "Use these only for choices the user confirmed in the conversation "
        "that are missing from their request below. Never add a feature, "
        "database, framework, tool or deployment step that is not in the "
        "user's request; where these notes disagree with it, the user's "
        "request wins.\n\n"
        f"{instructions}\n\n"
        + with_starting_point(
            f"{VERBATIM_MARKER}\n\n"
            "This is the authority. Where the summary above is shorter or "
            "differs, this text wins.\n\n"
            f"{original}",
            starting_point,
        )
    )


def starting_point(has_request: bool) -> str:
    """The run's starting point when the previous result is no longer available."""
    sources = (
        "the current model and the requests below; where they ask to continue or fix "
        "earlier work, build that work as part of the new application"
        if has_request else "the current model"
    )
    return (
        "The previous result is no longer available, so there is no earlier code to "
        f"continue or fix. Build the whole application from {sources}."
    )


def with_starting_point(text: str, point: str) -> str:
    """A "Starting point" section, when there is one, followed by ``text``: the
    user's newest message stays the end of the instructions."""
    if not point:
        return text
    return f"## Starting point\n\n{point}\n\n{text}" if text else f"## Starting point\n\n{point}"
