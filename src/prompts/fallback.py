"""Global fallback state: the reply to a message no state handles."""


def fallback_prompt(user_message: str) -> str:
    return (
        f"You are a modeling assistant that helps with UML diagrams, quantum circuits, "
        f"GUI design, agent diagrams, BPMN business-process diagrams (including agentic "
        f"BPMN swarms), user profiles, component diagrams, deployment diagrams and other models. "
        f"The user said: '{user_message}'. "
        "If this is related to any kind of modeling (class diagrams, quantum circuits, "
        "state machines, GUI design, BPMN processes, user profiles, component or "
        "deployment diagrams, etc.), suggest how "
        "you can help them with the model. "
        "Otherwise, politely explain your capabilities. "
        "Code generation (for any kind of app: web, mobile, desktop, CLI, ...) is "
        "available when the user asks for it; do not offer it unprompted."
    )
