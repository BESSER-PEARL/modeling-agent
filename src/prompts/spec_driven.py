"""Spec-Driven Agent hand-off: the instructions sent with ``trigger_smart_generator``."""


def framed_instructions(instructions: str, original: str, starting_point: str = "") -> str:
    """The classifier's summary, labelled as possibly inaccurate, then the
    user's verbatim request as the authority, then the run's starting point.
    An empty summary leaves out the notes."""
    if not instructions:
        return with_starting_point(f"## The user's original request, verbatim\n\n{original}", starting_point)
    return with_starting_point(
        "## Assistant's notes (may be inaccurate)\n\n"
        "Use these only for choices the user confirmed in the conversation "
        "that are missing from their request below. Never add a feature, "
        "database, framework, tool or deployment step that is not in the "
        "user's request; where these notes disagree with it, the user's "
        "request wins.\n\n"
        f"{instructions}\n\n"
        "## The user's original request, verbatim\n\n"
        "This is the authority. Where the summary above is shorter or "
        "differs, this text wins.\n\n"
        f"{original}",
        starting_point,
    )


def starting_point(has_request: bool) -> str:
    """The run's starting point when the previous result is no longer available."""
    sources = (
        "the current model and the requests above; where they ask to continue or fix "
        "earlier work, build that work as part of the new application"
        if has_request else "the current model"
    )
    return (
        "The previous result is no longer available, so there is no earlier code to "
        f"continue or fix. Build the whole application from {sources}."
    )


def with_starting_point(text: str, point: str) -> str:
    """``text`` followed by a "Starting point" section, when there is one."""
    if not point:
        return text
    return f"{text}\n\n## Starting point\n\n{point}" if text else f"## Starting point\n\n{point}"
