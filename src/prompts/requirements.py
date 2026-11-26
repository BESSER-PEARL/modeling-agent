"""Requirements extraction: the atomic requirements an application request states.

Adapted from the extraction rules of BESSER's requirements ledger
(``besser/spec_driven_agent/planning/requirements_ledger.py``), which judges the
same requirements against the generated code. Unlike the ledger, this list also
keeps the structure and the technology the user states, because the editor
traces every requirement to what realizes it, not only the behaviour a judge
can check.
"""

from common.agent_config import MAX_REQUIREMENTS

EXTRACTION_SYSTEM_PROMPT = (
    "You turn a user's application request into the list of atomic requirements it states.\n"
    "Rules:\n"
    "- One requirement per item, in the user's own words where possible.\n"
    "- List only what the request itself states. Never add what an application of that kind "
    "usually has: no validations, screens, roles, authentication or integrations the text does "
    "not name.\n"
    "- A vague request states little; return few items or an empty list.\n"
    "- quote: copy the words of the request that state the requirement EXACTLY, character for "
    "character: a phrase or one sentence, never reworded and never joined from several places.\n"
    "- data: one item per entity the user describes (with the attributes stated for it) and one per "
    "stated relationship between entities.\n"
    "- Where stated, also list: identifiers and unique values (uniqueness); the shape a value must "
    "have (validation); limits and rules across entities (rule); values worked out rather than "
    "typed in (computed); state changes and when they are refused (transition); named operations "
    "(action); screens and interactions (ui); external services such as payments or email "
    "(integration); performance, security, accessibility or other qualities of the whole "
    "application (quality); the technologies, platforms and architecture to use (technical).\n"
    "- Not a requirement: instructions to the assistant about the conversation itself (the language "
    "to answer in, how long to take, what to do next).\n"
    f"- At most {MAX_REQUIREMENTS} items. Never invent a requirement the user did not state."
)


def extraction_prompt(request_text: str) -> str:
    """The user prompt of the extraction call: the user's request, verbatim."""
    return f"## The user's request\n\n{request_text}"
