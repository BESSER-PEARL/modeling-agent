"""The container's config.yaml is byte-identical to what the old Dockerfile
heredoc wrote, and its log copy redacts the key the way the old ``sed`` did.

The expected texts are the old entrypoint's output, captured from an image
built from the old Dockerfile (Docker drops the heredoc's ``#`` lines).
"""

import pytest

from app import container_config
from common.settings import AgentSettings

DEFAULT_CONFIG = """\
agent:
  check_transitions_delay: 5

nlp:
  language: en
  region: US
  timezone: Europe/Madrid
  pre_processing: True
  intent_threshold: 0.55
  openai:
    api_key: {key}

platforms:
  websocket:
    host: 0.0.0.0
    port: 8765
    origins:
      - "{origin}"
      - "{origin_alt}"
      - "http://localhost:8080"
      - "http://localhost:5173"
      - "http://localhost:3000"
    streamlit:
      host: localhost
      port: 5000
"""


def _expected(key="", origin="https://editor.besser-pearl.org", origin_alt="https://experimental.besser-pearl.org"):
    return DEFAULT_CONFIG.format(key=key, origin=origin, origin_alt=origin_alt)


@pytest.fixture
def env(monkeypatch):
    for name in ("OPENAI_API_KEY", "BESSER_AGENT_WS_ORIGIN", "BESSER_AGENT_WS_ORIGIN_ALT"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_unset_environment_renders_the_defaults(env):
    assert container_config.render(AgentSettings()) == _expected()


def test_blank_values_fall_back_like_the_shell_default(env):
    for name in ("OPENAI_API_KEY", "BESSER_AGENT_WS_ORIGIN", "BESSER_AGENT_WS_ORIGIN_ALT"):
        env.setenv(name, "")
    assert container_config.render(AgentSettings()) == _expected()


def test_values_are_written_verbatim(env):
    env.setenv("OPENAI_API_KEY", 'sk-$HOME"x\\y api_key: z')
    env.setenv("BESSER_AGENT_WS_ORIGIN", " spaced ")
    env.setenv("BESSER_AGENT_WS_ORIGIN_ALT", "https://c.example/api_key: q")
    text = container_config.render(AgentSettings())
    assert text == _expected('sk-$HOME"x\\y api_key: z', " spaced ", "https://c.example/api_key: q")
    # sed "s/\(api_key:\).*/\1 [REDACTED]/": the first api_key: on each line, to its end.
    redacted = container_config.redact(text)
    assert "    api_key: [REDACTED]\n" in redacted
    assert '      - "https://c.example/api_key: [REDACTED]\n' in redacted
    assert redacted.count("[REDACTED]") == 2


def test_the_log_copy_never_shows_the_key(env):
    env.setenv("OPENAI_API_KEY", "sk-secret")
    text = container_config.render(AgentSettings())
    assert "sk-secret" in text
    assert "sk-secret" not in container_config.redact(text)
    assert "    api_key: [REDACTED]\n" in container_config.redact(text)


def test_main_writes_the_file_logs_it_and_execs_the_agent(env, tmp_path, monkeypatch, capsys):
    env.setenv("OPENAI_API_KEY", "sk-secret")
    path = tmp_path / "config.yaml"
    calls = []
    monkeypatch.setattr(container_config, "CONFIG_PATH", str(path))
    monkeypatch.setattr(container_config.os, "execvp", lambda *args: calls.append(args))

    container_config.main()

    assert path.read_bytes() == _expected("sk-secret").encode()
    assert capsys.readouterr().out == "✅ config.yaml created successfully\n" + _expected("[REDACTED]")
    assert calls == [("python", ["python", "modeling_agent.py"])]
