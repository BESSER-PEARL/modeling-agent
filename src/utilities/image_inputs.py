"""Images as LLM inputs: base64 data URLs to provider message content.

Callers pass images as ``data:<media type>;base64,<data>`` URLs, the form
OpenAI-compatible APIs accept directly; Anthropic takes the same bytes as
a base64 ``image`` block.
"""

from typing import Any, Dict, List, Sequence, Union


def data_url(mime_type: str, b64: str) -> str:
    return f"data:{mime_type};base64,{b64}"


def openai_content(text: str, images: Sequence[str] = ()) -> Union[str, List[Dict[str, Any]]]:
    """Chat Completions user content: the plain string without images."""
    if not images:
        return text
    return [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": url}} for url in images
    ]


def anthropic_content(text: str, images: Sequence[str] = ()) -> Union[str, List[Dict[str, Any]]]:
    """Messages API user content, images before the text as Anthropic advises."""
    if not images:
        return text
    blocks: List[Dict[str, Any]] = []
    for url in images:
        header, _, b64 = url.partition(",")
        media_type = header.removeprefix("data:").removesuffix(";base64")
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })
    blocks.append({"type": "text", "text": text})
    return blocks


def content_text(content: Union[str, List[Dict[str, Any]]]) -> str:
    """The text of a message content, for logging without image bytes."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")
