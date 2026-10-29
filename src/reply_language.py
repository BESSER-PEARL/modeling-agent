"""Replies in the user's language.

The classifier reports the language each message is written in
(``reply_language``, kept in ``REPLY_LANGUAGE``). LLM-authored replies are
asked to answer in it (:func:`language_instruction`); templated English replies
are translated at the reply choke point (``session_helpers``) by
:func:`localize_payload`: one classifier-tier call per reply, cached per
(text, language). Only the user-facing ``message`` and quick-action
``label``s are translated. A quick action's ``prompt`` is a machine token
matched by the confirm gates, and model / canvas payloads are data, so neither
is ever touched. On any failure the English text is sent.
"""

import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from session_keys import REPLY_LANGUAGE

logger = logging.getLogger(__name__)

_LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "lb": "Luxembourgish", "ca": "Catalan",
    "pl": "Polish", "ro": "Romanian", "el": "Greek", "tr": "Turkish", "ru": "Russian",
    "uk": "Ukrainian", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "cs": "Czech", "hu": "Hungarian", "ar": "Arabic", "he": "Hebrew",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
}

_CACHE_MAX = 1000
_cache: "OrderedDict[tuple, str]" = OrderedDict()
_cache_lock = threading.Lock()

# Markdown link targets, e.g. "(wme:add-key)", must survive translation as-is.
_LINK_TARGET_RE = re.compile(r"\]\(([^)\s]+)\)")

# Overridable in tests; defaults to the shared classifier-tier JSON call.
_predict_json: Optional[Callable[[str], str]] = None


def session_language(session: Any) -> Optional[str]:
    """The user's language code when it is not English, else None."""
    try:
        code = session.get(REPLY_LANGUAGE)
    except Exception:
        return None
    return code if isinstance(code, str) and code and code != "en" else None


def language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


def language_instruction(session: Any) -> str:
    """A "reply in <language>" line for LLM-authored replies, or ""."""
    code = session_language(session)
    if not code:
        return ""
    return (
        f"Reply in {language_name(code)}, the language the user writes in. Keep "
        "class, attribute and other model element names exactly as they are."
    )


def localize_payload(session: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """*payload* with its ``message`` and quick-action labels translated into
    the user's language. Returns *payload* itself when nothing changes."""
    code = session_language(session)
    if not code or not isinstance(payload, dict):
        return payload
    actions = payload.get("suggestedActions")
    actions = actions if isinstance(actions, list) else []
    texts: List[str] = []
    if isinstance(payload.get("message"), str) and payload["message"].strip():
        texts.append(payload["message"])
    for action in actions:
        if isinstance(action, dict) and isinstance(action.get("label"), str) and action["label"].strip():
            texts.append(action["label"])
    if not texts:
        return payload
    translated = dict(zip(texts, translate_texts(texts, code)))
    out = dict(payload)
    if isinstance(out.get("message"), str):
        out["message"] = translated.get(out["message"], out["message"])
    if actions:
        out["suggestedActions"] = [
            {**action, "label": translated.get(action.get("label"), action.get("label"))}
            if isinstance(action, dict) else action
            for action in actions
        ]
    return out


def translate_text(session: Any, text: str) -> str:
    code = session_language(session)
    if not code or not isinstance(text, str) or not text.strip():
        return text
    return translate_texts([text], code)[0]


def translate_texts(texts: List[str], code: str) -> List[str]:
    """Translate English *texts* into *code*; English on any failure."""
    with _cache_lock:
        missing = [t for t in dict.fromkeys(texts) if (t, code) not in _cache]
    if missing:
        try:
            for source, target in zip(missing, _translate_uncached(missing, code)):
                with _cache_lock:
                    _cache[(source, code)] = target
                    while len(_cache) > _CACHE_MAX:
                        _cache.popitem(last=False)
        except Exception as exc:
            logger.warning("Reply translation to %s failed; sending English: %s", code, exc)
            return list(texts)
    with _cache_lock:
        return [_cache.get((t, code), t) for t in texts]


def _translate_uncached(texts: List[str], code: str) -> List[str]:
    predict = _predict_json
    if predict is None:
        import agent_context as ctx
        predict = ctx.gpt_predict_json
    if predict is None:
        raise RuntimeError("no LLM available")
    prompt = (
        f"Translate each string in \"texts\" from English into {language_name(code)}. "
        "They are messages and button labels of a software-modeling assistant. Rules:\n"
        "- Keep markdown exactly: line breaks, bullets, **bold** and *italic* markers.\n"
        "- Keep unchanged: anything in backticks, link targets in (...) such as "
        "(wme:add-key), URLs, the product names BESSER and Spec-Driven Agent, and "
        "bold model element names that are identifiers (e.g. **Book**, **LoanItem**). "
        "Bold phrases of ordinary words are translated and stay bold.\n"
        "- Translate naturally; add nothing and drop nothing.\n"
        'Return JSON {"texts": [...]} with the same number of strings in the same order.\n\n'
        + json.dumps({"texts": texts}, ensure_ascii=False)
    )
    raw = predict(prompt)
    out = json.loads(raw).get("texts") if raw else None
    if not isinstance(out, list) or len(out) != len(texts):
        raise ValueError("translation returned a different number of strings")
    result = []
    for source, target in zip(texts, out):
        ok = isinstance(target, str) and target.strip() and (
            _LINK_TARGET_RE.findall(source) == _LINK_TARGET_RE.findall(target)
        ) and source.count("`") == target.count("`")
        if not ok:
            logger.warning("Reply translation dropped markup; keeping English for one string")
        result.append(target if ok else source)
    return result
