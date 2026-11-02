"""Conversation memory: the rolling-summary prompt."""


def summary_prompt(transcript: str) -> str:
    return (
        "Summarize this conversation history in 2-3 sentences. "
        "Focus on: what diagrams were created/modified, key decisions, "
        "and the user's current goal. Be concise.\n\n"
        f"{transcript}"
    )
