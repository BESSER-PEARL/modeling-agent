"""Central configuration for the modeling agent.

All tunable constants live here so they can be adjusted in one place
instead of being scattered across modules.
"""

# ── Tab / workspace limits ────────────────────────────────────────────────
MAX_TABS = 5

# ── Message handling ──────────────────────────────────────────────────────
MAX_USER_MESSAGE_CHARS = 64_000

# ── Requirements extraction ──────────────────────────────────────────────
# BESSER's MAX_REQUIREMENTS and MAX_REQUIREMENT_CHARS
# (spec_driven_agent/planning/specification.py): the Spec-Driven Agent rejects
# a request carrying more requirements, or a longer one.
MAX_REQUIREMENTS = 128
MAX_REQUIREMENT_CHARS = 1_000

# ── Traceability digest (context.traceDigest) ────────────────────────────
# The editor caps its digest (~12k characters serialized); the agent does not
# trust the client and caps every list and string again. List caps are the
# editor's (cross-repo contract, section 3); the nested ones bound one focus
# artifact. A string longer than its cap is cut, never rejected.
TRACE_DIGEST_MAX_RECENT_OPERATIONS = 15
TRACE_DIGEST_MAX_REQUIREMENTS = 40
TRACE_DIGEST_MAX_OUT_OF_DATE = 30
TRACE_DIGEST_MAX_GENERATIONS = 10
TRACE_DIGEST_MAX_FOCUS = 5
TRACE_DIGEST_MAX_FOCUS_HISTORY = 20
TRACE_DIGEST_MAX_FOCUS_DERIVED_FROM = 10
TRACE_DIGEST_MAX_FOCUS_DERIVED = 20
TRACE_DIGEST_MAX_LIST_ITEMS = 20       # a list of names (by, tools, realizedBy, reasons)
TRACE_DIGEST_MAX_NAME_CHARS = 200      # names, ids, paths, dates, kinds
TRACE_DIGEST_MAX_TEXT_CHARS = 500      # requirement text and quote, summaries, old/new values
# The largest count (counts.*, a generation's files, schemaVersion) taken as
# real: a bigger one is dropped, so a forged number cannot blow up the record
# or the classifier's workspace line (10**4000 renders to 4,000 digits).
TRACE_DIGEST_MAX_COUNT = 10_000_000
# The rendered record in the trace-question prompt (utilities.trace_digest):
# about 4k tokens, so the answer call stays on the small tier's budget.
TRACE_DIGEST_PROMPT_MAX_CHARS = 16_000
# The current-model summary a trace answer gets (requirement coverage needs
# the model's elements); cut so the record stays the subject.
TRACE_QA_PROJECT_SUMMARY_MAX_CHARS = 6_000

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
