FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    HOME=/home/besser

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && groupadd --gid 10001 besser \
    && useradd --uid 10001 --gid besser --create-home --home-dir /home/besser --shell /usr/sbin/nologin besser \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stock besser-agentic-framework 4.3.2 can evict a live connection when a
# second browser WebSocket using the same stable user ID closes. Keep this
# ownership-guarded vendored fix until it is available upstream.
COPY patches/websocket_platform.py \
    /usr/local/lib/python3.11/site-packages/baf/platforms/websocket/websocket_platform.py

COPY --chown=besser:besser . .

# Generate the BAF config at container start because it contains the provider
# key. Store it in ephemeral, owner-only space and never echo it to logs.
RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    'runtime_dir=/tmp/besser-modeling-agent' \
    'mkdir -p "$runtime_dir"' \
    'umask 077' \
    'cat > "$runtime_dir/config.yaml" <<EOF' \
    'agent:' \
    '  check_transitions_delay: 5' \
    '' \
    'nlp:' \
    '  language: en' \
    '  region: US' \
    '  timezone: Europe/Madrid' \
    '  pre_processing: True' \
    '  intent_threshold: 0.55' \
    '  openai:' \
    '    api_key: ${OPENAI_API_KEY:-}' \
    '' \
    'platforms:' \
    '  websocket:' \
    '    host: 0.0.0.0' \
    '    port: 8765' \
    '    streamlit:' \
    '      host: localhost' \
    '      port: 5000' \
    'EOF' \
    'cd "$runtime_dir"' \
    'exec python /app/modeling_agent.py' \
    > /usr/local/bin/besser-modeling-agent \
    && chmod 0755 /usr/local/bin/besser-modeling-agent

USER 10001:10001

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1', 8765), 5); s.close()" || exit 1

STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/besser-modeling-agent"]
