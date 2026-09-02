"""User-facing reply copy shared across modules.

Detection of these situations is the unified classifier's job (LLM-first);
this module only holds the words. It imports nothing, so both the state
bodies and the generation handler can use it without layering cycles.
"""

# The product name for the LLM-augmented generation feature. Every
# user-visible sentence uses THIS constant — the frontend already brands
# the run card "Spec-Driven Agent"; the chat replies must match.
SPEC_DRIVEN_NAME = "Spec-Driven Agent"


def continue_generating_prompt(artifact: str = "web app") -> str:
    """The single post-screens pause sentence (M5).

    Two emit paths used to carry two hand-written variants ("your web
    app" vs "your application"), which read as inconsistent polish and
    broke text-matching test harnesses. Every pause now goes through
    this one function.
    """
    return (
        "You can now review or refine your model, or continue with "
        f"generating your {artifact}. What would you like to do?"
    )


DECLINE_ACK = (
    "No problem — I'm here whenever you'd like to build or change something. "
    "Just tell me what you have in mind."
)

OUT_OF_SCOPE_REDIRECT = (
    "That's a bit outside what I do — I model **software systems** (class "
    "diagrams, state machines, BPMN, agents, and more) and generate code from "
    "them. What would you like to model? For example: *Create a library "
    "management system*."
)

META_ANSWER = (
    "Here's what I do: describe what you want in plain words and I turn it "
    "into a real, editable **model** (a class diagram you can see and refine "
    "on the canvas), then generate **consistent, runnable code** from it "
    "with BESSER's generators — a full web app (React + FastAPI), SQL, "
    "Django, Pydantic, SQLAlchemy, REST APIs, JSON Schema, and more. Unlike "
    "a general chatbot, the model stays your single source of truth: evolve "
    "it and regenerate any time instead of ending up with one-shot code "
    "that drifts. I also modify and describe diagrams, design state "
    "machines, BPMN processes, agents, and quantum circuits, and import "
    "PlantUML or diagram images.\n\nWhat would you like to build?"
)
