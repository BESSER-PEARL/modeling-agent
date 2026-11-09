"""The Docker entrypoint: write config.yaml from AgentSettings, then run the agent.

``python -m app.container_config`` writes ``/app/config.yaml``, logs it with
the ``api_key`` line redacted and replaces itself with ``python
modeling_agent.py``.
"""

from __future__ import annotations

import os
import re
import string
import sys

from common.settings import AgentSettings

CONFIG_PATH = "/app/config.yaml"

# platforms.websocket.origins is the CORS whitelist: BAF passes it to
# websockets.serve(origins=...), which accepts every origin when it is absent.
_TEMPLATE = string.Template("""\
agent:
  check_transitions_delay: 5

nlp:
  language: en
  region: US
  timezone: Europe/Madrid
  pre_processing: True
  intent_threshold: 0.55
  openai:
    api_key: ${openai_api_key}

platforms:
  websocket:
    host: 0.0.0.0
    port: 8765
    origins:
      - "${ws_origin}"
      - "${ws_origin_alt}"
      - "http://localhost:8080"
      - "http://localhost:5173"
      - "http://localhost:3000"
    streamlit:
      host: localhost
      port: 5000
""")

_API_KEY_RE = re.compile(r"(api_key:).*")


def render(settings: AgentSettings) -> str:
    """The container's config.yaml for *settings*."""
    return _TEMPLATE.substitute(
        openai_api_key=settings.openai_api_key.get_secret_value(),
        ws_origin=settings.ws_origin,
        ws_origin_alt=settings.ws_origin_alt,
    )


def redact(text: str) -> str:
    """*text* with the rest of every ``api_key:`` line replaced, for the log."""
    return _API_KEY_RE.sub(r"\1 [REDACTED]", text)


def main() -> None:
    text = render(AgentSettings())
    with open(CONFIG_PATH, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    print("✅ config.yaml created successfully")
    sys.stdout.write(redact(text))
    sys.stdout.flush()  # exec discards unflushed output
    os.execvp("python", ["python", "modeling_agent.py"])


if __name__ == "__main__":
    main()
