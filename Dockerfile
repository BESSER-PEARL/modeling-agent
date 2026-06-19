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
# unconditionally when a websocket closes. With our stable ?user_id= param
# (added so conversation memory survives reconnects), the two WS connections
# a browser tab opens — the assistant widget + the workspace drawer — share
# one session key, so a closing connection evicts the live one and the
# agent's replies are silently dropped. The vendored file adds an
# ownership-guarded delete + reply-route-to-sender. Remove this once the fix
# is upstreamed to BAF. Pinned to 4.3.2 — re-vendor if the version bumps.
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
    streamlit:\n\
      host: localhost\n\
      port: 5000\n\
EOF\n\
\n\
echo "✅ config.yaml created successfully"\n\
cat /app/config.yaml\n\
\n\
# Run the modeling agent\n\
exec python modeling_agent.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()" || exit 1

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
