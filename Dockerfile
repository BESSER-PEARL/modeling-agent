# Dockerfile for BESSER Modeling Agent
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Build behind a TLS-inspecting proxy: drop its root + signing certs into
# ca-certs-extra/ (gitignored) and pass --build-arg TRUST_EXTRA_CAS=1.
# Build-time only: the trust is removed again after the dependencies install.
ARG TRUST_EXTRA_CAS=0
COPY ca-certs-extra/ /tmp/ca-certs-extra/
RUN if [ "$TRUST_EXTRA_CAS" = "1" ]; then \
        cp /tmp/ca-certs-extra/*.crt /usr/local/share/ca-certificates/ \
        && update-ca-certificates; \
    fi; \
    rm -rf /tmp/ca-certs-extra
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

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
# Stock besser-agentic-framework 4.5.3 deletes the _connections slot
# unconditionally when a websocket closes; two connections sharing one session
# key (a stable ?user_id=) then evict each other and replies are silently
# dropped. It also keys a session by the connection id when no ?session_id= is
# sent, which the editor never does. The vendored file keys by ?user_id= and
# adds an ownership-guarded delete + reply routing to the sender. Remove once
# upstreamed; re-vendor if the BAF version bumps.
COPY patches/websocket_platform.py \
    /usr/local/lib/python3.11/site-packages/baf/platforms/websocket/websocket_platform.py

# Copy the modeling agent code and install it in place (editable: the code
# reads src/resources/ and uml_specs/ next to it). No index access: the
# dependencies are installed above and the base image ships setuptools.
COPY . .
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e .

# A build-time CA must not become runtime trust. Unconditional; the grep is an
# assertion that fails the build if a known TLS-inspection CA is still trusted.
RUN rm -f /usr/local/share/ca-certificates/*.crt /app/ca-certs-extra/*.crt \
    && update-ca-certificates --fresh >/dev/null 2>&1 \
    && ! grep -qi goskope /etc/ssl/certs/ca-certificates.crt

# Expose the websocket port
EXPOSE 8765

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
# start-period must cover the FULL boot, not just process start: BAF trains a
# NER model plus one intent classifier per state (~20s each, 10 states) and
# only then opens the WebSocket, which takes several minutes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()" || exit 1

# Write config.yaml from the environment (src/app/container_config.py), then
# exec the agent.
ENTRYPOINT ["python", "-m", "app.container_config"]
