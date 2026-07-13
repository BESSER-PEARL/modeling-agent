"""Unified message classifier — ONE LLM call replaces BAF's intent
classification + the smart-gen sub-router.

Before this module:
  * BAF's ``predict_intent`` fired on every message (LLM call #1),
    picking which *state* to transition to based on long ``description=``
    keyword blobs embedded in each intent declaration.
  * Inside ``generation_state``, ``classify_generation_request`` fired
    a second LLM call (#2) to pick smart vs deterministic and extract
    generator_type / refined_instructions.

This module collapses both into a single structured-output call that
returns EVERY field any downstream state body needs, cached per-message
so repeat transition conditions don't re-classify.

Architecture:

  1. ``classify_message(request, llm_provider)`` → ``UnifiedClassification``.
     One OpenAI call with a clean rule-based system prompt, forced
     structured output via Pydantic. Never raises — on any failure,
     returns a safe ``fallback_intent`` classification so the agent
     gracefully degrades to its own fallback body.

  2. ``get_or_classify(session, request, llm_provider)`` wraps that
     with a per-message cache. A BAF event's id is used as the cache
     key, so the first transition condition / state body to ask
     triggers the classification and everyone else reads the cached
     answer.

The schema is deliberately wide: it carries fields for generation
sub-routing AND diagram-creation targets in one object, so every state
body can read from the same source of truth.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from protocol.types import AssistantRequest
from session_keys import (
    UNIFIED_CLASSIFICATION,
    UNIFIED_CLASSIFICATION_EVENT_ID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------


# Top-level intent — mirrors the ``new_intent`` declarations in
# ``modeling_agent.py``. If a new state is added there, mirror it here.
_INTENT_NAMES = Literal[
    "hello_intent",
    "create_complete_system_intent",
    "modify_model_intent",
    "modeling_help_intent",
    "describe_model_intent",
    "uml_spec_intent",
    "generation_intent",
    # Catch-all — BAF's own fallback state body runs when nothing
    # matches. Routed to whatever state fallback the current state
    # has (typically modeling_help_state).
    "fallback_intent",
]


# Deterministic generators BESSER has built-in. Must stay in sync with
# ``generation_handler.GENERATOR_KEYWORDS`` and the TypeScript frontend.
_DETERMINISTIC_GENERATOR_TYPES = Literal[
    "django",
    "backend",
    "web_app",
    "sql",
    "sqlalchemy",
    "python",
    "java",
    "pydantic",
    "jsonschema",
    "smartdata",
    "agent",
    "qiskit",
    "rest_api",
    "rdf",
    "export",
    "deploy",
]


# Which diagram the user is talking about, for create_complete_system /
# modify_model / describe_model routes.
_TARGET_DIAGRAM_TYPES = Literal[
    "ClassDiagram",
    "ObjectDiagram",
    "StateMachineDiagram",
    "AgentDiagram",
    "GUINoCodeDiagram",
    "QuantumCircuitDiagram",
    "BPMN",
]


class UnifiedClassification(BaseModel):
    """Everything the state machine needs to route a message — in one shot."""

    intent: _INTENT_NAMES = Field(
        ...,
        description=(
            "State-level intent. Pick exactly one:\n"
            "  'hello_intent'                   — an ACTUAL greeting / "
            "small-talk FROM the user (e.g. 'hi', 'hello there', 'thanks'). "
            "A QUESTION that merely QUOTES a greeting word is NOT a greeting: "
            "'which intent handles the user saying hello', 'what happens when "
            "the user says hi', 'does my bot greet people' are questions "
            "ABOUT the model → describe_model_intent, not hello_intent.\n"
            "  'create_complete_system_intent'  — user wants a NEW diagram or "
            "complete system FROM SCRATCH (e.g. 'create a class diagram for a "
            "library', 'model a booking system', 'build a Grover algorithm "
            "circuit'). Also: adding an AGENT / CHATBOT / conversational "
            "assistant / bot to an app or project (set "
            "target_diagram_type='AgentDiagram') — that means a NEW agent "
            "diagram, not a class.\n"
            "  'modify_model_intent'            — user wants to ADD / REMOVE / "
            "CHANGE elements in an EXISTING diagram (e.g. 'add a class called "
            "User', 'remove the Book class', 'connect Author and Book'). This "
            "ALSO covers INDIRECT requests that express a wish for the model "
            "to CAPTURE / TRACK / STORE / RECORD / REMEMBER a piece of DATA "
            "about an existing entity — e.g. 'it would help to know when each "
            "order was placed' (→ add a date attribute to Order), 'I'd like "
            "to keep track of each customer's address', 'we should remember "
            "the shipping date', 'every product needs a price'. The user is "
            "asking you to MODEL that data, not asking a question — route "
            "modify_model_intent and let the modify step add the "
            "attribute/relationship. BUT a QUESTION asking for your advice "
            "about WHETHER or HOW to change the model ('do I need an Address "
            "class?', 'how would you improve this?', 'what should I add?') is "
            "NOT a modify command — it is describe_model_intent (analyze + "
            "advise). Route modify_model_intent only for a STATEMENT/COMMAND "
            "of a concrete change ('add a price to Product', 'every product "
            "needs a price', 'connect Customer and Order'), never for an "
            "open question seeking your recommendation. ALSO NOT a model "
            "edit: after a recent smart/vibe generation, a request to add a "
            "FEATURE to the generated app/code ('add authentication to it', "
            "'add a login page', 'make it responsive') is generation_intent "
            "(smart), not modify_model_intent — see the SMART-GEN FOLLOW-UP "
            "rule in the system prompt.\n"
            "  'describe_model_intent'          — user is asking ABOUT their "
            "current diagram, wanting to KNOW/SEE what ALREADY exists (e.g. "
            "'what classes do I have?', 'list all states', 'is X connected to "
            "Y?'). This ALSO covers EVALUATIVE / ADVISORY questions that ask "
            "for your OPINION, RECOMMENDATION, or CRITIQUE of the existing "
            "model — 'is my model any good?', 'how would you improve this "
            "design?', 'what would you add?', 'what am I missing?', 'do I "
            "need a separate Address class?', 'are there problems with my "
            "design?', 'what should I change?', 'is this a good way to model "
            "X?'. These ask you to ANALYZE and ADVISE (and you SHOULD answer "
            "with grounded, specific suggestions about THEIR classes) — they "
            "are NOT commands to make a change, so do NOT route them to "
            "modify_model_intent and do NOT set needs_clarification. NOT a "
            "request to add new data — if the user wants the model to start "
            "holding a NEW piece of information, that is modify_model_intent.\n"
            "  'modeling_help_intent'           — user asks for CONCEPTUAL "
            "help (e.g. 'how do I model inheritance?', 'explain UML "
            "composition')\n"
            "  'uml_spec_intent'                — user asks about the formal "
            "UML specification\n"
            "  'generation_intent'              — user wants SOURCE CODE in ANY "
            "language or stack, or to EXPORT / DEPLOY. Includes BESSER built-ins "
            "(django, pydantic, sql, ...) AND any other language (rails, rust, "
            "kotlin, next.js, go, ...). A request to CREATE/design a NEW model "
            "AND generate code from it in one message (e.g. 'create a booking "
            "system and generate django') is create_complete_system_intent, "
            "NOT generation_intent — build the model first; the agent offers to "
            "generate afterward. After a recent smart/vibe generation, a "
            "follow-up asking to add/change a FEATURE of the generated "
            "app/code ('add auth to it', 'add a dashboard to the app', 'make "
            "it responsive') is generation_intent with "
            "generation_route='smart' (reuse_for_generation) — the frontend "
            "re-runs the vibe generator in incremental modify mode.\n"
            "  'fallback_intent'                — none of the above fit cleanly."
        ),
    )

    # --- Generation-only fields (populated when intent == 'generation_intent') ---

    generation_route: Optional[Literal["smart", "deterministic", "modeling", "other"]] = Field(
        default=None,
        description=(
            "REQUIRED when intent='generation_intent'. Sub-routing:\n"
            "  'deterministic' — user wants ONE BESSER built-in with NO extras "
            "(auth, JWT, Docker, migrations, …). Pure scaffolding.\n"
            "  'smart' — user wants a non-BESSER stack (rails, rust, kotlin, "
            "...) OR a BESSER built-in PLUS extras the template can't produce "
            "(auth, JWT, OAuth, Docker, custom DB, migrations, tests, rate-"
            "limiting, custom middleware).\n"
            "  'modeling' — this is actually a 'generate a diagram' request "
            "misrouted here (use this to redirect).\n"
            "  'other' — not a code-generation request at all."
        ),
    )
    generator_type: Optional[_DETERMINISTIC_GENERATOR_TYPES] = Field(
        default=None,
        description=(
            "REQUIRED when generation_route='deterministic'. Name of the "
            "BESSER built-in generator to run."
        ),
    )
    refined_instructions: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when generation_route='smart'. A polished prompt for the "
            "smart generator naming the stack (e.g. 'Rails 7, PostgreSQL via "
            "Active Record, Devise auth') and any non-functional requirements "
            "the user mentioned. Max 2000 chars. Do NOT describe the class "
            "diagram in detail — the generator has the domain model. Do NOT "
            "invent requirements the user didn't mention."
        ),
    )
    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description=(
            "Suggested LLM provider when generation_route='smart'. Ignored "
            "for other routes. The frontend's BYOK dropdown can override."
        ),
    )

    # --- Domain-mismatch fields (populated when generation_route='smart') ---
    # Used to refuse silent code-rewrites when the user's request describes
    # a different domain than their existing class diagram.

    domain_mismatch: Optional[bool] = Field(
        default=None,
        description=(
            "ONLY when generation_route='smart' AND a class diagram with at "
            "least one class is present in WORKSPACE CONTEXT. True if the "
            "user's request describes a domain that DOES NOT match the "
            "existing class diagram (e.g. classes are 'Team/Player' but the "
            "request is 'a shoe store'). False if the request fits the "
            "existing model OR the model is empty/absent. Be conservative: "
            "if unsure, return False. Leave NULL when route != 'smart' or "
            "when there's no existing class diagram to compare against."
        ),
    )
    suggested_new_domain: Optional[str] = Field(
        default=None,
        description=(
            "When domain_mismatch=True, a SHORT noun phrase naming the "
            "domain the user actually wants (e.g. 'a shoe store', 'a hotel "
            "booking system', 'a blog platform'). Used in the agent's "
            "follow-up question. Max 80 chars. Leave NULL otherwise."
        ),
    )

    # --- Modeling-side fields (create / modify / describe) ---

    target_diagram_type: Optional[_TARGET_DIAGRAM_TYPES] = Field(
        default=None,
        description=(
            "For create_complete_system_intent / modify_model_intent / "
            "describe_model_intent — which diagram the user is talking "
            "about. Leave NULL if the user didn't specify and the "
            "active diagram in the workspace context should be used."
        ),
    )

    model_disposition: Optional[Literal[
        "extend_existing",
        "replace_existing",
        "new_tab",
        "reuse_for_generation",
        "new_from_scratch",
    ]] = Field(
        default=None,
        description=(
            "How the request relates to the EXISTING workspace model — judge "
            "from WORKSPACE CONTEXT (does a usable model already exist?) plus "
            "the wording. Populate for create / modify / generation "
            "intents:\n"
            "  'extend_existing'      — add to / change the current model "
            "('add a Payment class', 'also include returns', 'expand this', "
            "'add auth to my model').\n"
            "  'replace_existing'     — user EXPLICITLY wants to discard the "
            "current model and start over ('scrap this and start fresh with', "
            "'replace it with a ...', 'redo it as').\n"
            "  'new_tab'              — build a SEPARATE new diagram alongside "
            "('in a new tab', 'a second diagram for', 'also model a ...').\n"
            "  'reuse_for_generation' — generate code / an app FROM the "
            "existing model WITHOUT changing it ('generate X from my model', "
            "'build an app from this', 'from my class diagram', 'use what I "
            "have', 'the diagram I've been working on').\n"
            "  'new_from_scratch'     — no usable model exists yet, OR the "
            "user clearly describes a brand-new system to build from nothing.\n"
            "Leave NULL ONLY when genuinely ambiguous — the agent will ask "
            "rather than guess destructively."
        ),
    )

    needs_clarification: bool = Field(
        default=False,
        description=(
            "Set True ONLY when acting would require a GUESS: no resolvable "
            "referent ('do the thing', 'make it better', 'fix it'), an "
            "unresolved pronoun with no antecedent ('rename it', 'connect "
            "them' with nothing prior to point at), two materially different "
            "readings, or a missing target the WORKSPACE CONTEXT cannot "
            "supply. Do NOT set it for requests you can reasonably act on, "
            "for plain confirmations ('yes'/'no'), or merely because a "
            "request is broad — bias toward acting when a sensible default "
            "exists. When True, also write clarifying_question."
        ),
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when needs_clarification=True. ONE short, specific "
            "question that, once answered, lets you act — reference the "
            "workspace where useful (e.g. 'Which class should I add the email "
            "attribute to — Customer or Order?'). Max 200 chars. LEAVE NULL "
            "otherwise."
        ),
    )

    reason: str = Field(
        ...,
        description=(
            "One short sentence (max 160 chars) explaining the classification. "
            "Used for logs and surfaced to users as a hint."
        ),
    )


_SYSTEM_PROMPT = (
    "You are an intent classifier. Classify the user's message into "
    "one of the listed intents and, when relevant, populate the "
    "sub-routing fields. Return the structured classification only — "
    "no prose, no questions, no follow-ups.\n\n"
    "=== TOP-LEVEL INTENT RULES (pick one) ===\n\n"
    "hello_intent: greetings, small-talk, capability questions "
    "('what can you do'), thanks, acknowledgements. A question about "
    "the user's OWN model or app is NEVER hello — in particular "
    "'where is the app?', 'how do I run / try / see / use it?', "
    "'can I try it?' are generation_intent, not hello.\n\n"
    "create_complete_system_intent: user wants a NEW diagram or "
    "complete system from scratch. Keywords that trigger this: "
    "'create a class diagram for', 'design a system', 'model a', "
    "'generate a class diagram', 'build a state machine for', "
    "'create the GUI for'. If they name a domain ('library', "
    "'e-commerce', 'hotel booking') and ask for a diagram or system, "
    "it's this. CRITICAL: 'generate a class diagram' is this intent, "
    "NOT generation_intent — they want a DIAGRAM, not source code.\n"
    "CRITICAL (AGENT / CHATBOT): a request to ADD or CREATE an AGENT, "
    "CHATBOT, conversational assistant, virtual assistant, or bot — "
    "whether 'to the app', 'to my project', 'to the website', or "
    "standalone — is create_complete_system_intent with "
    "target_diagram_type='AgentDiagram'. An agent/chatbot is its OWN "
    "conversational diagram (states + intents), NOT a class on the class "
    "diagram and NOT a GUI element. This holds NO MATTER which diagram is "
    "currently active (class diagram, GUI, etc.). Examples: 'add a "
    "chatbot to the web app that answers navigation questions', 'I want "
    "an agent for my app', 'add a conversational assistant', 'build a "
    "support bot' → ALL create_complete_system_intent + "
    "target_diagram_type='AgentDiagram'. Do NOT classify these as "
    "modify_model_intent just because the user said 'add ... to the "
    "app'.\n\n"
    "modify_model_intent: user wants to ADD / REMOVE / CHANGE "
    "elements in an existing diagram. 'add a class', 'remove the "
    "Book class', 'rename', 'delete', 'connect', 'add an attribute', "
    "'modify method', 'I also want to include', 'extend with', 'add "
    "a gate to the circuit'. Also single-element creation: 'create "
    "a class called User', 'make a state'. EXCEPTION: adding an "
    "agent / chatbot / conversational assistant / bot is NOT this "
    "intent — see the AGENT / CHATBOT rule above (it means a new "
    "AgentDiagram). But a class that merely HAS the word 'agent' in "
    "its name (e.g. 'create a class called AgentManager', 'add an "
    "Agent class with a name attribute') IS a class modification — "
    "keep it modify_model_intent on the class diagram. EXCEPTION 2 "
    "(smart-gen follow-up): if the RECENT CONVERSATION shows a "
    "just-completed smart / Spec-Driven generation and the user asks to "
    "ADD or CHANGE a FEATURE of the generated app/code ('add "
    "authentication to it', 'add a login page', 'make it responsive', "
    "'add a dashboard to the app'), that is generation_intent (smart), "
    "NOT a model edit — see the SMART-GEN FOLLOW-UP rule below. Route "
    "here only when the user changes an actual class / attribute / "
    "relationship of the DOMAIN MODEL.\n\n"
    "describe_model_intent: user asks QUESTIONS about their CURRENT "
    "diagram. 'how many classes', 'what attributes', 'list all', "
    "'tell me about my model', 'describe', 'summarize', 'what does "
    "this circuit do'. Always about what ALREADY EXISTS. EXCEPTION: a "
    "question asking to SEE / PREVIEW the model AS CODE in a specific "
    "target language or format ('what does this look like as postgres "
    "sql?', 'show me this as SQL', 'spit out the postgres create "
    "statements', 'what would the Django models look like?') is "
    "generation_intent, NOT describe_model_intent — even though phrased "
    "as a question, the user wants the generator's OUTPUT in that "
    "format, not a description of the classes/attributes.\n\n"
    "modeling_help_intent: conceptual help, explanations, best "
    "practices. 'how do I', 'explain', 'what is', 'how does X work', "
    "'what are best practices for'. Conceptual, not about their "
    "specific model. ALSO: how to RUN / START / SET UP code the user "
    "ALREADY generated or downloaded — 'I downloaded the zip, how do I "
    "run it?', 'how do I start the generated backend?', 'how do I run "
    "the app on my laptop?'. They HAVE the code and need setup/run "
    "steps, not a new generation.\n\n"
    "uml_spec_intent: asks about the formal UML specification "
    "document. 'according to UML spec', 'what does UML standard say', "
    "'OMG specification'. Rare.\n\n"
    "generation_intent: user wants SOURCE CODE, EXPORT, or DEPLOY. "
    "Includes BESSER's built-in generators (django, pydantic, sql, "
    "sqlalchemy, python, java, web_app, backend, jsonschema, "
    "smartdata, agent, qiskit, rest_api, rdf) AND ANY OTHER language "
    "or framework (ruby on rails, rust, kotlin, swift, go, elixir, "
    "c#, c++, php, laravel, flask, express, next.js, spring boot, "
    "angular, vue, svelte, ios, android). Also: export to json/buml "
    "('export as json', 'save the project to json'), and DEPLOY — "
    "'deploy to render', 'deploy this model', 'push this to prod', "
    "'go ahead and deploy it', 'ship it to production' are ALL "
    "generation_intent wanting the deploy action, even without the "
    "literal word 'deploy'/'render'. ALSO includes asking to RUN, TRY, PREVIEW, "
    "LAUNCH, USE, or SEE the app, or 'where is the app?' when they do "
    "NOT yet have generated code — the user has a model and wants "
    "runnable output, which comes from generating code. BUT if they "
    "ALREADY have generated or downloaded code (they mention a zip, "
    "the downloaded app, the generated backend/code) and ask how to "
    "RUN it, that is modeling_help_intent, not generation. NEVER use "
    "this when the user says 'generate a class diagram' — that's "
    "create_complete_system_intent.\n\n"
    "CREATE-vs-GENERATE (important): a request to GENERATE code / an app "
    "FROM AN EXISTING or current model — 'generate a dashboard from my "
    "class diagram', 'build a react + fastapi app from my model', "
    "'generate a webapp' when a model already exists — is "
    "generation_intent. A request to CREATE / design a NEW model AND "
    "generate from it in ONE message — 'create a booking system and "
    "generate django', 'design a library and build it end-to-end' — is "
    "create_complete_system_intent: the agent builds the model first, "
    "then offers to generate the code. Do NOT invent a separate "
    "end-to-end intent.\n\n"
    "SMART-GEN FOLLOW-UP (add/change a FEATURE of the just-generated "
    "app) — READ THIS BEFORE choosing modify_model_intent: when the "
    "RECENT CONVERSATION shows the agent JUST ran the smart / Spec-Driven "
    "generator (a turn mentioning 'Spec-Driven Agent', 'Smart "
    "generation', or '[smart-generation outcome]') AND the new message "
    "asks to ADD or CHANGE a FEATURE of the GENERATED APP / CODE, that is "
    "generation_intent with generation_route='smart' and "
    "model_disposition='reuse_for_generation' — NOT modify_model_intent. "
    "The frontend re-runs the Vibe generator in incremental modify mode "
    "on the SAME project, so it edits the GENERATED CODE, never the class "
    "diagram. Signals: app/feature words plus a pronoun pointing at the "
    "app ('it', 'the app', 'the code', 'the generated app', 'the site', "
    "'the page'). Examples right after a smart gen: 'add authentication "
    "to it' / 'add a login system to it' → generation_intent (smart); "
    "'add a login page' / 'add a dashboard to the app' / 'add a search "
    "bar' → generation_intent (smart); 'make it responsive' / 'add dark "
    "mode' / 'protect the admin routes' → generation_intent (smart). "
    "CRITICAL BUG THIS PREVENTS: 'add a authentication system to it' "
    "after a smart gen must NOT become modify_model_intent (which would "
    "wrongly add a UserAccountSystem CLASS to the diagram) — the user "
    "wants auth in the GENERATED CODE. DISCRIMINATOR: route here ONLY "
    "when the user changes what the GENERATED APP does or looks like. A "
    "genuine DOMAIN-MODEL edit that names a class, attribute, or "
    "relationship ('add a Payment class', 'add an email attribute to "
    "Member', 'rename Book to Publication', 'remove the Loan class', "
    "'connect Order to Customer') stays modify_model_intent EVEN right "
    "after a generation. If there was NO recent smart generation in the "
    "conversation, classify 'add X' by its normal meaning.\n\n"
    "fallback_intent: none of the above fits cleanly.\n\n"
    "=== GENERATION SUB-ROUTING (populate when intent='generation_intent') ===\n\n"
    "CRITICAL BACKGROUND: BESSER has two generation paths.\n"
    "  * 'deterministic' = pure scaffolding from a template. No auth, "
    "no JWT, no OAuth, no Docker, no migrations, no tests — JUST the "
    "baseline. If the user wants ANY extra feature, deterministic is "
    "wrong.\n"
    "  * 'smart' = scaffolding + custom features. Internally runs a "
    "deterministic template first, then the LLM adds custom features "
    "on top.\n\n"
    "Sub-routing — apply the PRINCIPLE first, use the examples only as "
    "illustrations (they are NOT an exhaustive keyword list):\n"
    "PRINCIPLE: route 'smart' whenever the request needs ANY custom "
    "behavior, business rule, access control, role/permission, "
    "authentication, login/signup, integration, deployment target, a "
    "specific/real UI or dashboard, non-default infrastructure, OR a "
    "non-BESSER language/framework — anything a bare CRUD template "
    "cannot express. Route 'deterministic' ONLY for the plain baseline "
    "scaffold of a single named BESSER generator with NO added "
    "behavior. Decide on the MEANING of the request, NOT on whether a "
    "specific adjective/stack keyword appears. When genuinely torn "
    "between smart and deterministic, prefer 'smart' — it produces a "
    "real working app, whereas deterministic only emits an empty "
    "scaffold. These are ALL 'smart': 'a web app for managing my "
    "inventory', 'a site where customers can browse and order', 'make "
    "it production-ready', 'only admins can edit records', 'let users "
    "log in', 'vibe-code me something cool from my model', 'turn this "
    "into a dashboard'. A FEATURE added to an app the smart generator "
    "just built (see SMART-GEN FOLLOW-UP above: 'add auth to it', 'add a "
    "dashboard to the app', 'make it responsive') is ALWAYS 'smart'.\n"
    "Illustrative examples:\n"
    "1. Non-BESSER language/framework (rails, rust, kotlin, swift, go, "
    "elixir, php, laravel, flask, express, next.js, spring boot, "
    "angular, vue, svelte, ios/android app, ...) → 'smart'.\n"
    "2. Compound build ('backend + frontend', 'react + fastapi', "
    "'full-stack fastapi with jwt + postgres', 'dockerized next.js') → "
    "'smart'.\n"
    "2b. A DASHBOARD, or a real/custom web app, website, or application "
    "described by what it should DO or look like → 'smart'. BESSER's "
    "deterministic 'web_app' generator only emits a generic CRUD GUI "
    "scaffold and needs a GUI diagram; reserve generator_type='web_app' "
    "for when the user EXPLICITLY asks to run BESSER's web_app / GUI "
    "generator. 'smart' examples: 'generate me a full dashboard', "
    "'build me a webapp from my model', 'make a react dashboard'.\n"
    "3. ANY BESSER built-in PLUS custom behavior or features — "
    "authentication, login/signup, roles / permissions / access-control "
    "('only admins can edit', 'users see only their own data'), JWT, "
    "OAuth, Docker/containers, a specific non-default database, "
    "migrations, tests, rate-limiting, middleware, CORS, CI/CD, "
    "business rules, or integrations → 'smart'. Examples: 'web app with "
    "authentication', 'django with jwt', 'backend that runs in a "
    "container'.\n"
    "3b. EXCEPTION — a named SQL DIALECT is NOT an 'extra': when the "
    "request is for the SQL or SQLAlchemy generator specifically and "
    "simply NAMES the target dialect/DBMS (postgres/postgresql, mysql, "
    "sqlite, mssql, mariadb, oracle) with NO other added behavior, that "
    "is the generator's own required parameter — NOT the 'specific "
    "non-default database' extra from rule 3. Stay 'deterministic' with "
    "generator_type='sql' (or 'sqlalchemy') and let the dialect be "
    "parsed from the message; do NOT escalate to 'smart' just because a "
    "dialect is named. Examples (all 'deterministic', generator_type='sql'): "
    "'what does this look like as postgres sql?', 'spit out the postgres "
    "create statements', 'generate mysql for my model'. Only escalate when "
    "the user ALSO asks for something the template can't do (auth, Docker, "
    "business rules, a full app around the schema, ...).\n"
    "4. A bare BESSER built-in with NO added behavior → 'deterministic' "
    "with generator_type set. The BESSER built-ins, each with its OWN "
    "deterministic template, are: django, backend, sql, sqlalchemy, "
    "python, java, pydantic, jsonschema, smartdata, web_app, agent, "
    "qiskit, rest_api, rdf (plus export, deploy). When the user names ONE of these and "
    "asks for nothing extra, route 'deterministic' and set generator_type "
    "to it. The decoration words 'code', 'models', 'classes', 'schema', "
    "'from my model', 'for my diagram' do NOT make it smart — they are "
    "just how people ask for the plain output. Map the obvious phrasings: "
    "'generate java' / 'java classes' / 'java code from my model' → "
    "generator_type='java'; 'json schema' / 'generate jsonschema' → "
    "'jsonschema'; 'sqlalchemy' / 'sqlalchemy models' → 'sqlalchemy'; "
    "'pydantic' / 'pydantic models' → 'pydantic'; 'python' / 'python "
    "classes' (plain domain model, no app) → 'python'; 'sql' → 'sql'; "
    "'django' → 'django'; 'rest api' → 'rest_api'; 'rdf' → 'rdf'; "
    "'smartdata' → 'smartdata'. NOTE: 'smartdata' is "
    "a BESSER generator NAME — despite containing the substring 'smart', "
    "naming it ('run smartdata on this model', 'generate smartdata', "
    "'smartdata for my crm model') is ALWAYS 'deterministic' with "
    "generator_type='smartdata', NEVER the 'smart' route; do not let the "
    "substring match fool you. Also: naming the SPECIFIC EXISTING classes "
    "from the loaded model in the request ('write the pydantic schema for "
    "Customer and Order', 'create the sqlalchemy orm for me', 'give me "
    "the pydantic model for Order') does NOT make it 'smart' and is NEVER "
    "an invitation for YOU to hand-author code inline — it is a plain "
    "request to RUN the named deterministic generator on the existing "
    "model. Route 'deterministic' with the matching generator_type; never "
    "invent fields, never ask the user to redescribe classes that are "
    "already listed in WORKSPACE CONTEXT. Only escalate ONE of "
    "these to 'smart' when the user ALSO asks for added behavior (auth, "
    "JWT, Docker, a custom DB, migrations, tests, a real/custom app or "
    "dashboard) or a NON-BESSER stack. Examples (all 'deterministic'): "
    "'generate django', 'give me pydantic classes', 'generate sql', "
    "'generate java code from my model', 'generate a json schema', "
    "'generate sqlalchemy models', 'generate python code'.\n"
    "4c. EXPORT vs jsonschema: 'export my project', 'export/save/download "
    "as json', 'export the model', 'download my project' mean the "
    "project-FILE export → generator_type='export', NOT 'jsonschema'. "
    "Reserve generator_type='jsonschema' for an explicit 'json schema' / "
    "'JSON Schema' request that wants a schema DOCUMENT describing the "
    "model. The bare word 'json' inside an EXPORT request never means "
    "jsonschema.\n"
    "4d. DEPLOY vs EXPORT: judge the VERB, not the noun that follows it. "
    "'export' / 'save' / 'download' + (json/buml/project/model) → "
    "generator_type='export'. 'deploy' / 'publish' / 'push ... to prod' / "
    "'ship ... to production' / 'go live' / 'launch it' → "
    "generator_type='deploy', generation_route='deterministic' — "
    "REGARDLESS of whether the sentence also contains the word 'model' "
    "or 'project'. 'deploy this model', 'push this to prod', 'go ahead "
    "and deploy it', 'ship this to production' are ALL "
    "generator_type='deploy', NEVER 'export' just because they mention "
    "'model'/'project'. A bare deploy/publish request needs NO extra "
    "config to still count as 'deterministic' — BESSER's deploy flow "
    "collects hosting/repo details in its own dialog afterward, so a "
    "deploy request is ALWAYS 'deterministic' with generator_type='deploy', "
    "never 'smart' and never 'other', even if it also mentions Docker, "
    "CI/CD, or a specific host.\n"
    "4b. Vague 'how do I run / try / see / get the app' or 'where is "
    "the app' with NO stack named → 'deterministic' with "
    "generator_type=null (the agent then shows the generator menu so "
    "the user picks what to build).\n"
    "5. User actually wants a DIAGRAM (not source code) → 'modeling'.\n"
    "6. Greetings / small-talk leaking through → 'other'.\n\n"
    "=== domain_mismatch (populate when generation_route='smart') ===\n"
    "If WORKSPACE CONTEXT lists CLASS NAMES from an existing class "
    "diagram, judge whether those classes describe the SAME DOMAIN as "
    "the user's request:\n"
    "  * Classes 'Team', 'Player' + request 'build a shoe store webapp' "
    "→ domain_mismatch=True, suggested_new_domain='a shoe store'.\n"
    "  * Classes 'Book', 'Author' + request 'add JWT auth and Docker' "
    "→ domain_mismatch=False (request is about stack, not domain).\n"
    "  * Classes 'Customer', 'Order' + request 'generate a rust rest "
    "api' → domain_mismatch=False. A request that names only a "
    "LANGUAGE / FRAMEWORK / STACK / API style (rust, go, kotlin, spring "
    "boot, react, fastapi, 'a rest api', 'a graphql api', 'a "
    "microservice', 'a backend', 'a web app') has NO business domain of "
    "its own — it inherits the existing model's domain, so it is NEVER a "
    "mismatch. Only a request naming a DIFFERENT BUSINESS DOMAIN (shoe "
    "store, hotel, hospital, blog, airline) can be a mismatch.\n"
    "  * No class names present (empty model or non-class diagram only) "
    "→ domain_mismatch=null.\n"
    "BE CONSERVATIVE: if the request could be applied on top of the "
    "existing classes, return False. Only flag True when the request "
    "names a clearly DIFFERENT business domain than the existing "
    "classes — never merely because it names a stack or framework.\n\n"
    "=== refined_instructions (populate when generation_route='smart') ===\n"
    "A polished, implementation-focused prompt for the smart generator:\n"
    "  * name the stack explicitly (Rails, PostgreSQL, Devise auth, ...)\n"
    "  * include non-functional requirements the user mentioned\n"
    "  * max 2000 chars\n"
    "  * do NOT describe the class diagram in detail (the generator "
    "    has the domain model)\n"
    "  * for a SMART-GEN FOLLOW-UP (adding a feature to the app the "
    "    generator just built) describe ONLY the feature to ADD (e.g. "
    "    'Add user authentication: login/signup, session handling, "
    "    protect existing routes') — the generator re-runs incrementally "
    "    over the existing codebase, so do NOT re-describe the whole app "
    "    or the domain model\n"
    "  * do NOT invent requirements the user didn't mention\n\n"
    "=== target_diagram_type (create / modify / describe) ===\n"
    "Which diagram the user is talking about. Leave NULL if they "
    "didn't specify — the state body will use the workspace's active "
    "diagram. Set it when they explicitly say 'the class diagram', "
    "'my state machine', 'the quantum circuit', etc. ALWAYS set "
    "target_diagram_type='AgentDiagram' when the user asks to add or "
    "create an agent / chatbot / conversational assistant / bot, even "
    "if they are currently viewing the class diagram or GUI — the new "
    "agent belongs in its own AgentDiagram.\n\n"
    "=== model_disposition (use WORKSPACE CONTEXT) ===\n"
    "Read WORKSPACE CONTEXT to see what already exists, then say how the "
    "request relates to it. 'reuse_for_generation' = generate code/an app "
    "from the existing model without changing it ('build an app from my "
    "model', 'from my class diagram', 'use what I have'). Adding a "
    "FEATURE to an app the smart generator just built ('add auth to it', "
    "'make it responsive' right after a smart gen) is ALSO "
    "'reuse_for_generation' — the domain model itself is not changed. "
    "'extend_existing' = add to / modify the current model. "
    "'replace_existing' = explicitly discard and start over. 'new_tab' = a "
    "separate new diagram alongside. 'new_from_scratch' = nothing usable "
    "exists yet or a brand-new unrelated system. Do NOT default to a "
    "destructive rebuild just because the wording omits 'my model' — if a "
    "usable model is present and the user asks to generate from it, that is "
    "'reuse_for_generation'. Leave NULL only when truly ambiguous.\n\n"
    "=== needs_clarification (ask instead of guess) ===\n"
    "Set needs_clarification=True ONLY when acting would require a GUESS: a "
    "referent-less request ('do the thing', 'make it better', 'fix it'), an "
    "unresolved pronoun with no antecedent ('rename it', 'connect them' with "
    "nothing prior), two materially different readings, or a missing target "
    "the WORKSPACE CONTEXT cannot supply. Then write ONE short, specific "
    "clarifying_question. Do NOT ask when a reasonable default exists, for "
    "plain 'yes'/'no' confirmations, or merely because a request is broad — "
    "bias strongly toward acting; only ask when genuinely stuck.\n\n"
    "=== OUTPUT ===\n"
    "Return the structured classification. Always include 'reason' "
    "(≤160 chars) explaining your choice. Be decisive — do not "
    "second-guess; the state machine trusts your verdict."
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def classify_message(
    request: AssistantRequest,
    llm_provider: Any,
    history: Optional[list] = None,
) -> UnifiedClassification:
    """Classify a user message into a state-level intent + sub-routing fields.

    ONE classifier-tier structured-output call (see ``model_config``).
    Returns a safe
    ``fallback_intent`` classification if the provider is unavailable
    or the call fails — the caller should trust the returned object
    and dispatch based on ``intent``.

    ``history`` is an optional list of prior ``{"role", "content"}`` turns
    (most recent last) so the classifier can resolve referents like "the
    same", "continue", "it", or "do that for all" against what was just
    asked/done — this is what makes the agent feel like it remembers the
    session instead of treating every message cold.

    Never raises — the classifier must never crash the agent.
    """
    if llm_provider is None:
        return _safe_fallback("LLM provider unavailable")
    message = (request.message or "").strip()
    if not message:
        return _safe_fallback("empty message")

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_block(request, history)},
        ]
        # Reasoning models (gpt-5* / o-series) burn hidden reasoning tokens
        # from the SAME completion budget. With only 800 tokens the visible
        # structured output is starved → parsed=None → _safe_fallback on
        # EVERY message (#44). Give reasoning models headroom; keep the tight
        # budget for fast non-reasoning models like gpt-4o-mini.
        from model_config import supports_custom_temperature

        classifier_model = getattr(llm_provider, "model_name", "") or ""
        max_tokens = 800 if supports_custom_temperature(classifier_model) else 4000
        result: UnifiedClassification = llm_provider.parse(
            messages=messages,
            schema=UnifiedClassification,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if result is None:
            return _safe_fallback("LLM returned no result")
        return _post_validate(result, message)
    except Exception:
        logger.exception("classify_message failed; falling back to fallback_intent")
        return _safe_fallback("LLM classifier failed")


def get_or_classify(
    session: Any,
    request: AssistantRequest,
    llm_provider: Any,
) -> UnifiedClassification:
    """Per-message cache wrapper around :func:`classify_message`.

    Uses the BAF event's id as the cache key. The first caller on a
    given message triggers the classification; subsequent callers on
    the SAME message read the cached result without an extra LLM call.

    Every transition condition and state body on a single incoming
    WebSocket message should go through this helper so the whole
    request consumes exactly ONE classification call.
    """
    event_id = _current_event_id(session)
    cached_event_id = session.get(UNIFIED_CLASSIFICATION_EVENT_ID)
    cached_classification = session.get(UNIFIED_CLASSIFICATION)
    if (
        event_id is not None
        and cached_event_id == event_id
        and cached_classification is not None
    ):
        return cached_classification

    # Frontend callbacks (``generator_result`` etc.) are protocol events,
    # not user prose — their routing is determined by the ``action``
    # field, so classifying their text is pure waste AND wrong: in
    # production a generation-completion echo was LLM-classified as
    # ``hello_intent`` and routed to greetings, so the generation
    # handler's frontend_event branch never ran.
    if getattr(request, "action", None) == "frontend_event":
        result = UnifiedClassification(
            intent="generation_intent",
            generation_route="other",
            reason="frontend_event callback — routed deterministically, no LLM call",
        )
    else:
        result = classify_message(
            request, llm_provider, _recent_history(session, request)
        )
        _log_classification(request, result)
    if event_id is not None:
        session.set(UNIFIED_CLASSIFICATION, result)
        session.set(UNIFIED_CLASSIFICATION_EVENT_ID, event_id)
    return result


def _log_classification(
    request: AssistantRequest, result: UnifiedClassification
) -> None:
    """One concise INFO line per real classification — the agent's routing
    decision. The only previously-invisible step in the pipeline; makes
    "why did it route there?" answerable from the logs. Never raises."""
    try:
        msg = (request.message or "").strip().replace("\n", " ")
        if len(msg) > 80:
            msg = msg[:77] + "..."
        bits = [f"intent={result.intent}"]
        if result.generation_route:
            bits.append(f"route={result.generation_route}")
        if result.generator_type:
            bits.append(f"gen={result.generator_type}")
        if result.target_diagram_type:
            bits.append(f"target={result.target_diagram_type}")
        if result.model_disposition:
            bits.append(f"disp={result.model_disposition}")
        if result.domain_mismatch:
            bits.append("domain_mismatch=True")
        if result.needs_clarification:
            bits.append("clarify=True")
        logger.info("[classify] %s | %s", " ".join(bits), msg)
    except Exception:  # pragma: no cover - logging must never break routing
        pass


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# How many prior turns to feed the classifier, and how much of each.
_HISTORY_MAX_TURNS = 6
_HISTORY_MAX_CHARS = 240


def _recent_history(session: Any, request: AssistantRequest) -> Optional[list]:
    """Best-effort: the last few conversation turns for referent resolution.

    Read at classification time — which is BEFORE ``_common_preamble``
    records the *current* user message — so this returns the PRIOR turns
    only, exactly the context needed to resolve referents in the current
    message ("the same", "continue", "do it for all", ...).

    Lazy import keeps ``unified_classifier`` importable without the memory
    stack and sidesteps any import cycle. Never raises.
    """
    try:
        from memory import get_memory, memory_session_key

        mem = get_memory(memory_session_key(session, request))
        return mem.get_last_n(_HISTORY_MAX_TURNS)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("recent_history unavailable (best-effort): %s", exc)
        return None


def _history_lines(history: Optional[list]) -> list:
    """Render prior turns as compact ``role: content`` lines (oldest first)."""
    if not history:
        return []
    out: list = []
    for turn in history[-_HISTORY_MAX_TURNS:]:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or "?").strip()
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > _HISTORY_MAX_CHARS:
            content = content[: _HISTORY_MAX_CHARS - 3].rstrip() + "..."
        out.append(f"{role}: {content}")
    return out


def _build_user_block(
    request: AssistantRequest, history: Optional[list] = None
) -> str:
    """Compose the user message + recent conversation + workspace context."""
    lines = ["USER MESSAGE:", request.message or ""]

    # Recent conversation turns so the classifier can resolve referents
    # ("the same", "continue", "do that for all", "it", "that one")
    # against what was just asked/done — the difference between an agent
    # that remembers the session and one that treats every message cold.
    hist_lines = _history_lines(history)
    if hist_lines:
        lines.append("")
        lines.append(
            "RECENT CONVERSATION (oldest first; resolve referents such as "
            '"the same", "continue", "it", "do that for all" against it, '
            "but classify the USER MESSAGE above — not these prior turns):"
        )
        lines.extend(hist_lines)

    ctx = getattr(request, "context", None)
    if ctx is None:
        return "\n".join(lines)

    summary_lines = []
    active_type = getattr(ctx, "active_diagram_type", None)
    if active_type:
        summary_lines.append(f"- active diagram: {active_type}")
    # FULL editor content: every relevant diagram type with its element
    # count + a few element names, and an explicit "not present" line for
    # missing ones — so the classifier can reason about what already exists
    # and about prerequisites (e.g. a webapp request when no GUI exists).
    summary_lines.extend(_workspace_summary_lines(ctx))
    # Full class-name list (up to 30) for the domain_mismatch judgement.
    class_names = _extract_class_names(ctx)
    if class_names:
        summary_lines.append(
            "- all class names: " + ", ".join(class_names[:30])
        )

    if summary_lines:
        lines.append("")
        lines.append("WORKSPACE CONTEXT (what is currently in the editor):")
        lines.extend(summary_lines)
    return "\n".join(lines)


# Sub-element types whose names are member rows, not top-level diagram
# elements — excluded from the per-diagram name preview.
_NAME_SKIP_TYPES = {
    "ClassAttribute", "ClassMethod", "ClassOCLConstraint",
    "ObjectAttribute", "ObjectMethod",
}

# The diagram types worth surfacing to the classifier, with a human unit.
_RELEVANT_DIAGRAM_TYPES = [
    ("ClassDiagram", "class(es)"),
    ("ObjectDiagram", "object(s)"),
    ("StateMachineDiagram", "state(s)"),
    ("AgentDiagram", "agent element(s)"),
    ("GUINoCodeDiagram", "GUI element(s)"),
    ("QuantumCircuitDiagram", "quantum element(s)"),
    ("BPMN", "process element(s)"),
]


def _diagram_element_names(model: Any, limit: int = 8) -> tuple[int, list[str]]:
    """(top-level element count, up to ``limit`` element names) for a model.

    Skips member rows (attributes/methods) and child elements so the names
    are the meaningful top-level entities. Never raises.
    """
    if not isinstance(model, dict):
        return 0, []
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return 0, []
    count = 0
    names: list[str] = []
    for elem in elements.values():
        if not isinstance(elem, dict):
            continue
        if elem.get("type") in _NAME_SKIP_TYPES:
            continue
        if elem.get("owner"):
            continue
        count += 1
        name = elem.get("name")
        if isinstance(name, str) and name.strip() and len(names) < limit:
            names.append(name.strip())
    return count, names


def _workspace_summary_lines(ctx: Any) -> list[str]:
    """Per-diagram-type summary of everything currently in the editor.

    One line per relevant diagram type: present types show their element
    count + a few names (across tabs); missing types show "not present".
    Never raises.
    """
    out: list[str] = []
    for dtype, unit in _RELEVANT_DIAGRAM_TYPES:
        try:
            diagrams = ctx.get_all_diagrams_of_type(dtype)
        except Exception:
            diagrams = []
        total = 0
        names: list[str] = []
        for d in diagrams:
            if not isinstance(d, dict):
                continue
            c, n = _diagram_element_names(d.get("model"))
            total += c
            for name in n:
                if name not in names and len(names) < 8:
                    names.append(name)
        if total > 0:
            tabs = f" across {len(diagrams)} tabs" if len(diagrams) > 1 else ""
            shown = ", ".join(names)
            more = f", +{total - len(names)} more" if names and total > len(names) else ""
            detail = f" ({shown}{more})" if shown else ""
            out.append(f"- {dtype}: {total} {unit}{tabs}{detail}")
        else:
            out.append(f"- {dtype}: not present")
    return out


def _extract_class_names(ctx: Any) -> list[str]:
    """Pull class names from the active ClassDiagram in the project snapshot.

    Returns an empty list when there is no ClassDiagram, the diagram is
    empty, or the snapshot shape is unexpected. Never raises.
    """
    try:
        diagram = ctx.get_diagram_from_snapshot("ClassDiagram")
    except Exception:
        return []
    if not isinstance(diagram, dict):
        return []
    model = diagram.get("model")
    if not isinstance(model, dict):
        return []
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return []
    names: list[str] = []
    for elem in elements.values():
        if not isinstance(elem, dict):
            continue
        if elem.get("type") not in ("Class", "AbstractClass"):
            continue
        name = elem.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


# Words that unambiguously signal a DEPLOY request in the safety nets
# below. Checked alongside "no export/save/download wording" so we never
# override a message that could genuinely mean export.
_DEPLOY_WORDS = ("deploy", "publish", "push", "ship", "go live", "launch")
_EXPORT_WORDS = ("export", "save", "download")


def _looks_like_unambiguous_deploy(lower_message: str) -> bool:
    """True when *lower_message* clearly means DEPLOY, not EXPORT.

    Used by :func:`_post_validate` to repair the classic mix-up where a
    deploy-shaped message ("deploy this model", "push this to prod") is
    mis-typed as ``generator_type='export'`` (structurally similar to
    "export the model") or dropped into the ``'other'`` (refusal) route.
    """
    return any(w in lower_message for w in _DEPLOY_WORDS) and not any(
        w in lower_message for w in _EXPORT_WORDS
    )


def _post_validate(result: UnifiedClassification, message: str = "") -> UnifiedClassification:
    """Defensive validation of LLM output.

    Catches the classic "LLM returned route='smart' but forgot to write
    instructions" and "intent='generation' but no generation_route"
    bugs. Repairs them in place rather than collapsing a valid intent to a
    worse one.
    """
    if result.intent == "generation_intent":
        if result.generation_route is None:
            logger.warning(
                "LLM returned generation_intent with no generation_route; "
                "demoting to fallback_intent"
            )
            return UnifiedClassification(
                intent="fallback_intent",
                reason="classifier missed generation sub-routing",
            )
        if result.generation_route == "smart":
            lower_msg = (message or "").lower()
            # Safety net: "smartdata" is a deterministic BESSER built-in
            # NAME that happens to contain the substring "smart" — the
            # LLM sometimes lexically confuses that with the 'smart'
            # (spec-driven) route. Only correct it when the message names
            # smartdata AND doesn't also ask for something the template
            # can't do (those genuinely belong on the smart path).
            _smart_extras = (
                "auth", "jwt", "oauth", "docker", "container", "migrat",
                "rate limit", "middleware", "cors", "ci/cd", "role",
                "permission", "login", "signup", "sign-in",
            )
            if re.search(r"\bsmart\s*data\b", lower_msg) and not any(
                w in lower_msg for w in _smart_extras
            ):
                logger.warning(
                    "LLM routed 'smartdata' request to smart via substring "
                    "confusion; correcting to deterministic/smartdata"
                )
                result.generation_route = "deterministic"
                result.generator_type = "smartdata"
            elif not (result.refined_instructions or "").strip():
                # Keep the smart/vibe route — do NOT collapse to the
                # deterministic generator menu, which silently kills requests
                # like "dashboard pls" or "vibe-code me something cool from my
                # model". Fall back to the raw user message as the generator
                # instructions so the smart path still runs.
                logger.warning(
                    "LLM returned smart route with no refined_instructions; "
                    "synthesizing instructions from the user message"
                )
                result.refined_instructions = (
                    (message or "").strip()
                    or "Build a custom application from the current model."
                )
        elif result.generation_route == "deterministic":
            # Safety net: a DEPLOY-shaped message ("deploy this model",
            # "push this to prod") sometimes gets mis-typed as 'export'
            # (structurally similar to "export the model") or left with
            # generator_type=None (falls through to the generic menu).
            lower_msg = (message or "").lower()
            if result.generator_type in (None, "export") and _looks_like_unambiguous_deploy(
                lower_msg
            ):
                logger.warning(
                    "LLM routed a deploy-shaped message to generator_type=%s; "
                    "correcting to deploy",
                    result.generator_type,
                )
                result.generator_type = "deploy"
            # Otherwise generator_type may legitimately be None — the
            # caller will show the generator menu. That's fine; no
            # demotion needed.
        elif result.generation_route == "other":
            # Safety net: an unambiguous deploy request ("go ahead and
            # deploy it", "push this to prod") sometimes gets classified
            # as 'other' (no clear code-generation request) and the agent
            # refuses instead of deploying. Promote it the same way as
            # the 'deterministic' branch above.
            lower_msg = (message or "").lower()
            if _looks_like_unambiguous_deploy(lower_msg):
                logger.warning(
                    "LLM routed a deploy-shaped message to 'other'; "
                    "correcting to deterministic/deploy"
                )
                result.generation_route = "deterministic"
                result.generator_type = "deploy"
    return result


def _safe_fallback(reason: str) -> UnifiedClassification:
    """Safe default when the LLM is unavailable.

    Returns ``fallback_intent`` so the agent runs its existing
    fallback state body. This is the pre-LLM-classifier behaviour —
    user gets a helpful message asking them to rephrase.
    """
    return UnifiedClassification(intent="fallback_intent", reason=reason)


def _current_event_id(session: Any) -> Optional[str]:
    """Best-effort event id used as the per-message cache key.

    BAF exposes the current event on ``session.event``; we try a few
    common attribute names and fall back to ``None`` (which disables
    caching — safe, just costs an extra LLM call if multiple callers
    on the same message ask independently).
    """
    event = getattr(session, "event", None)
    if event is None:
        return None
    for attr in ("id", "event_id", "uid"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    # Fall back to the raw event's id() — stable for the lifetime of
    # the event object, which is exactly one message dispatch.
    try:
        return f"obj:{id(event)}"
    except Exception:
        return None
