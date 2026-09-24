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

# ── LLM models ───────────────────────────────────────────────────────────
# The model name is referenced from a dozen call sites (agent setup, handlers,
# session helpers, the streaming provider, the file-conversion vision calls).
# It lives here so a migration is a change to these two constants rather than a
# find/replace across the tree — a blanket replace previously downgraded the
# vision calls, which need the full tier, to a mini model.
#
# When changing either value, add the matching entry to _COST_PER_1K in
# tracking/token_tracker.py; an unknown model falls back to placeholder pricing
# and every reported cost silently becomes an estimate.
LLM_MODEL_DEFAULT = "gpt-4.1-mini"
# Image -> model conversion. Deliberately the full tier, not the mini.
LLM_MODEL_VISION = "gpt-4.1"

# ── LLM parameters ───────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.2
LLM_TEXT_TEMPERATURE = 0.4
LLM_MAX_TOKENS_LARGE = 8192
LLM_MAX_TOKENS_SMALL = 2048
LLM_MAX_TOKENS_TEXT = 4096

# ── Conversation context ─────────────────────────────────────────────────
CONVERSATION_HISTORY_DEPTH = 5
