"""Model operations: the framing around the request a diagram handler receives."""


def spec_block(spec: str) -> str:
    """The user's verbatim spec, ahead of the planner's focused instruction."""
    return (
        "## The user's request, verbatim - THIS IS THE AUTHORITY\n\n"
        "Model exactly what it states: every status value it lists, every "
        "action it names, every rule it gives, every attribute it "
        "describes. The focused instruction that follows is a planner "
        "summary of this text, not a replacement for it - where they "
        "differ, this wins.\n\n"
        f"{spec}\n\n"
        "## Focused instruction\n\n"
    )
