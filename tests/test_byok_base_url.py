"""BYOK custom base_url (PIA / local providers) + SSRF gate.

A custom base_url makes the agent open a user-supplied URL — an SSRF surface on
a shared host — so it is OFF by default and only honoured when the deploy sets
BESSER_AGENT_ALLOW_CUSTOM_BASE_URL. When off, BYOK falls back to the shared LLM.
"""
import byok


def test_base_url_disabled_by_default(monkeypatch):
    monkeypatch.delenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", raising=False)
    token = byok.set_current("openai", "sk-test", "gpt-4o", "http://localhost:11434/v1")
    try:
        # Gate off → BYOK disabled entirely (request uses the shared LLM).
        assert byok.get_current() is None
    finally:
        byok.reset_current(token)


def test_base_url_honoured_when_flag_on(monkeypatch):
    monkeypatch.setenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", "1")
    token = byok.set_current("openai", "sk-test", "gpt-4o", "http://localhost:11434/v1")
    try:
        cfg = byok.get_current()
        assert cfg is not None
        assert cfg.base_url == "http://localhost:11434/v1"
        assert cfg.provider == "openai"
    finally:
        byok.reset_current(token)


def test_no_base_url_is_unaffected(monkeypatch):
    # A normal BYOK key (no custom endpoint) works regardless of the flag.
    monkeypatch.delenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", raising=False)
    token = byok.set_current("openai", "sk-test", "gpt-4o")
    try:
        cfg = byok.get_current()
        assert cfg is not None
        assert cfg.base_url is None
    finally:
        byok.reset_current(token)


def test_redacted_never_leaks_key_but_shows_base_url():
    cfg = byok.BYOKConfig(
        provider="openai", api_key="sk-secret", model="gpt-4o",
        base_url="http://localhost:11434/v1",
    )
    red = cfg.redacted()
    assert "sk-secret" not in red
    assert "localhost:11434" in red
