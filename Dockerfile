# Dockerfile for BESSER Modeling Agent
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Apply vendored BAF fix on top of the pip-installed framework.
# Stock besser-agentic-framework 4.3.2 deletes the _connections slot
# unconditionally when a websocket closes; two connections sharing one session
# key (a stable ?user_id=) then evict each other and replies are silently
# dropped. The vendored file adds an ownership-guarded delete + reply routing
# to the sender. Remove once upstreamed; re-vendor if the BAF version bumps.
COPY patches/websocket_platform.py \
    /usr/local/lib/python3.11/site-packages/baf/platforms/websocket/websocket_platform.py

# Copy the modeling agent code
COPY . .

# Expose the websocket port
EXPOSE 8765

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app

# Create entrypoint script that generates config.yaml from environment variables
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Generate config.yaml from environment variables\n\
cat > /app/config.yaml << EOF\n\
agent:\n\
  check_transitions_delay: 5\n\
\n\
nlp:\n\
  language: en\n\
  region: US\n\
  timezone: Europe/Madrid\n\
  pre_processing: True\n\
  intent_threshold: 0.55\n\
  openai:\n\
    api_key: ${OPENAI_API_KEY:-}\n\
\n\
platforms:\n\
  websocket:\n\
    host: 0.0.0.0\n\
    port: 8765\n\
    # CORS. BAF passes this to websockets.serve(origins=...); when the key is\n\
    # ABSENT every origin is accepted. Override the two host entries per\n\
    # deployment; the localhost entries are for local development.\n\
    origins:\n\
      - "${BESSER_AGENT_WS_ORIGIN:-https://editor.besser-pearl.org}"\n\
      - "${BESSER_AGENT_WS_ORIGIN_ALT:-https://experimental.besser-pearl.org}"\n\
      - "http://localhost:8080"\n\
      - "http://localhost:5173"\n\
      - "http://localhost:3000"\n\
    streamlit:\n\
      host: localhost\n\
      port: 5000\n\
EOF\n\
\n\
echo "✅ config.yaml created successfully"\n\
# Print the config for debugging but NEVER the API key: redact the\n\
# api_key line so the OpenAI key does not land in the container logs.\n\
sed "s/\\(api_key:\\).*/\\1 [REDACTED]/" /app/config.yaml\n\
\n\
# Run the modeling agent\n\
exec python modeling_agent.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
# start-period must cover the FULL boot, not just process start: BAF trains a
# NER model plus one intent classifier per state (~20s each, 10 states) and
# only then opens the WebSocket, which takes several minutes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()" || exit 1

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
