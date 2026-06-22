"""Central configuration for the modeling agent.

All tunable constants live here so they can be adjusted in one place
instead of being scattered across modules.
"""

# ── Tab / workspace limits ────────────────────────────────────────────────
MAX_TABS = 5

# ── Message handling ──────────────────────────────────────────────────────
MAX_USER_MESSAGE_CHARS = 12_000

# ── Session cleanup ──────────────────────────────────────────────────────
GRACE_PERIOD_SECONDS = 300

# ── Streaming ─────────────────────────────────────────────────────────────
STREAM_BUFFER_THRESHOLD = 200

# ── LLM parameters ───────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.2
LLM_TEXT_TEMPERATURE = 0.4
LLM_MAX_TOKENS_LARGE = 8192
LLM_MAX_TOKENS_SMALL = 2048
LLM_MAX_TOKENS_TEXT = 4096

# ── Conversation context ─────────────────────────────────────────────────
# How many recent messages are fed to the LLM verbatim each turn. The rolling
# summary (memory/conversation_memory.py) covers everything older, so the agent
# remembers the whole session, not just this window.
CONVERSATION_HISTORY_DEPTH = 10
