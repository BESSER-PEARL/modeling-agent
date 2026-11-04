"""Spec-Driven Agent hand-off: the instructions sent with ``trigger_smart_generator``."""


def framed_instructions(instructions: str, original: str) -> str:
    """The classifier's summary, labelled as possibly inaccurate, then the
    user's verbatim request as the authority."""
    return (
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
        f"{original}"
    )
