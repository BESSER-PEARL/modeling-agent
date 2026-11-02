"""A shared reject-not-truncate boundary for authoritative user requirements."""

from agent_config import MAX_USER_MESSAGE_CHARS


class UserMessageTooLong(ValueError):
    """The complete request cannot be accepted without losing requirements."""


def validate_message_length(message: str, *, label: str = "Your message") -> str:
    if len(message) > MAX_USER_MESSAGE_CHARS:
        raise UserMessageTooLong(
            f"{label} is too long ({len(message):,} characters; maximum "
            f"{MAX_USER_MESSAGE_CHARS:,}). Nothing was truncated or generated. "
            "Please shorten the specification before trying again."
        )
    return message
