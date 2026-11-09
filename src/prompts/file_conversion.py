"""File upload conversion: the JSON schema per diagram type and the prompts
that convert PlantUML, knowledge graphs, XMI, text, PDFs and images."""

CLASS_DIAGRAM_SCHEMA = """{
  "diagramType": "ClassDiagram",
  "classes": [
    {
      "className": "ClassName",
      "attributes": [
        {"name": "attrName", "type": "String", "visibility": "public"}
      ],
      "methods": [
        {"name": "methodName", "returnType": "void", "visibility": "public", "parameters": [
          {"name": "paramName", "type": "String"}
        ]}
      ]
    }
  ],
  "relationships": [
    {
      "type": "association|inheritance|composition|aggregation",
      "source": "SourceClassName",
      "target": "TargetClassName",
      "name": "relationshipName",
      "sourceMultiplicity": "1",
      "targetMultiplicity": "*"
    }
  ]
}
For inheritance, source is the subclass and target the superclass. For composition and aggregation, source is the PART and target the WHOLE (the class at the diamond: PlantUML "Brand *-- BrandURL" is source BrandURL, target Brand)."""

STATE_MACHINE_SCHEMA = """{
  "diagramType": "StateMachineDiagram",
  "states": [
    {
      "stateName": "StateName",
      "stateType": "initial|final|regular",
      "entryAction": "",
      "exitAction": "",
      "doActivity": ""
    }
  ],
  "transitions": [
    {
      "source": "StateA",
      "target": "StateB",
      "trigger": "event",
      "guard": "condition",
      "effect": "action"
    }
  ]
}"""

OBJECT_DIAGRAM_SCHEMA = """{
  "diagramType": "ObjectDiagram",
  "objects": [
    {
      "objectName": "object1",
      "className": "ClassName",
      "attributes": [
        {"name": "attrName", "value": "actualValue"}
      ]
    }
  ],
  "links": [
    {
      "source": "object1",
      "target": "object2",
      "relationshipType": "association"
    }
  ]
}"""

AGENT_DIAGRAM_SCHEMA = """{
  "diagramType": "AgentDiagram",
  "intents": [
    {
      "intentName": "IntentName",
      "trainingPhrases": ["phrase1", "phrase2"]
    }
  ],
  "states": [
    {
      "stateName": "stateName",
      "type": "state",
      "replies": [
        {"text": "Reply text", "replyType": "text"}
      ]
    }
  ],
  "transitions": [
    {
      "source": "initial",
      "target": "stateName",
      "condition": "when_intent_matched",
      "conditionValue": "IntentName"
    }
  ]
}"""

BPMN_SCHEMA = """{
  "diagramType": "BPMN",
  "nodes": [
    {"id": "start", "name": "Order Placed", "type": "startEvent"},
    {"id": "check_stock", "name": "Check Stock", "type": "task", "taskType": "service"},
    {"id": "gw", "name": "In stock?", "type": "gateway", "gatewayType": "exclusive"},
    {"id": "ship", "name": "Ship Order", "type": "task", "taskType": "user"},
    {"id": "end", "name": "Order Shipped", "type": "endEvent"}
  ],
  "flows": [
    {"source": "start", "target": "check_stock", "name": ""},
    {"source": "check_stock", "target": "gw", "name": ""},
    {"source": "gw", "target": "ship", "name": "yes"}
  ],
  "pools": [],
  "dataObjects": [{"id": "order", "name": "Order"}],
  "dataAssociations": [{"dataObject": "order", "node": "check_stock", "direction": "input"}]
}"""
# node "type" is one of: startEvent, endEvent, intermediateEvent, task, gateway.
# "taskType" (for task nodes) is one of: default, user, service, send, receive,
# manual, business-rule, script. "gatewayType" (for gateway nodes) is one of:
# exclusive, parallel, inclusive, event-based, complex.
# "pools" is optional — only populate it when the process names 2+ distinct
# participants/organizations (pool) or roles within one organization (pool
# lanes); each pool is {"id", "name", "lanes": [{"id", "name"}]} and referenced
# from a node via optional "poolId"/"laneId". Leave pools empty otherwise.
# "dataObjects" lists the data, documents or materials a step uses or produces;
# "dataAssociations" links each one to its node ("direction": input or output).
# Leave both empty when the text names none.

ALL_SCHEMAS = f"""
=== ClassDiagram (for entity/data models, schemas, class structures) ===
{CLASS_DIAGRAM_SCHEMA}

=== BPMN (for business process narratives: ordered steps/tasks, decisions, actors) ===
{BPMN_SCHEMA}

=== StateMachineDiagram (for object/system lifecycles: named states and transitions) ===
{STATE_MACHINE_SCHEMA}

=== ObjectDiagram (for specific instances/examples with concrete attribute values) ===
{OBJECT_DIAGRAM_SCHEMA}

=== AgentDiagram (for chatbot agents, intent/response flows, conversational AI) ===
{AGENT_DIAGRAM_SCHEMA}
"""


# Appended to the auto-detect prompt for an XMI file, before its content.
XMI_NOTE = (
    "IMPORTANT: This is an XMI (XML Metadata Interchange) file — a standard "
    "format for exchanging UML models. Parse the XML elements and attributes "
    "to extract classes, properties, associations, state machines, etc.\n\n"
)


def conversion_prompt(source_label: str, target_diagram_type: str) -> str:
    """Build a diagram-type-specific conversion prompt."""
    schema = _get_schema_for_type(target_diagram_type)
    type_label = diagram_type_label(target_diagram_type)

    return (
        f"You are a UML/modeling expert. Convert the given {source_label} "
        f"into a {type_label} specification.\n\n"
        f"Return ONLY a JSON object with this exact structure:\n{schema}\n\n"
        "IMPORTANT RULES:\n"
        "1. Extract ALL relevant elements from the source content\n"
        "2. Use proper type names: String, int, boolean, double, Date, float, long\n"
        "3. Return ONLY valid JSON, no explanations or markdown\n"
        "4. The \"diagramType\" field MUST be included and set correctly\n"
    )


def auto_detect_prompt(source_label: str) -> str:
    """Build a prompt that lets the LLM choose the best diagram type."""
    return (
        f"You are a UML/modeling expert. Analyze the given {source_label} and "
        "determine which diagram type best represents its content.\n\n"
        "Choose the BEST-FIT diagram type and return a JSON object using the matching schema:\n"
        f"{ALL_SCHEMAS}\n\n"
        "DECISION GUIDE (read carefully before choosing):\n"
        "- ClassDiagram: data models, entity structures, schemas, classes with attributes/methods, "
        "requirements documents, user stories, epics, feature lists, system specifications, "
        "any document describing WHAT a system manages (entities, roles, data). "
        "When in doubt between ClassDiagram and something entity-shaped, prefer ClassDiagram.\n"
        "- BPMN: a business/organizational PROCESS narrated as an ordered sequence of steps/tasks "
        "performed by one or more actors, with decision points ('if X then Y else Z'), parallel "
        "work, or hand-offs between actors (e.g., 'the customer places an order, then the system "
        "checks stock, then the warehouse prepares the package...'). This is the correct choice "
        "for narrative process/workflow descriptions — prefer it whenever the content reads as a "
        "sequence of activities/tasks rather than a formal state-transition list.\n"
        "- StateMachineDiagram: ONLY when the content explicitly defines named states and "
        "transitions between them (e.g., 'Idle -> Running -> Stopped'), describing the LIFECYCLE "
        "of a single object/system. Do NOT choose this for narrative process descriptions — "
        "those are BPMN, even if the document also mentions 'workflow' or 'process'.\n"
        "- ObjectDiagram: specific instances with concrete values (e.g., 'user1 : User, name=\"Alice\"')\n"
        "- AgentDiagram: chatbot flows, intent-response patterns, conversational AI\n\n"
        "IMPORTANT RULES:\n"
        "1. You MUST include the \"diagramType\" field in your response\n"
        "2. Extract ALL relevant elements from the source content\n"
        "3. Use proper type names: String, int, boolean, double, Date, float, long\n"
        "4. Return ONLY valid JSON, no explanations or markdown\n"
    )


def image_prompt(target_diagram_type: str) -> str:
    """Vision prompt for an image of a known diagram type."""
    schema = _get_schema_for_type(target_diagram_type)
    type_label = diagram_type_label(target_diagram_type)
    return (
        f"You are a UML/modeling expert. Analyze this image of a {type_label} "
        f"and convert it into a structured JSON specification.\n\n"
        f"Return ONLY a JSON object with this exact structure:\n{schema}\n\n"
        "IMPORTANT: Extract EVERY element visible in the diagram. "
        "Keep names exactly as shown. Include the \"diagramType\" field. "
        "Return ONLY valid JSON."
    )


def auto_detect_image_prompt(detect_mockup: bool, mockup_type: str) -> str:
    """Vision prompt that detects the diagram type; with ``detect_mockup`` it may
    answer ``{"diagramType": mockup_type}`` for a user interface instead."""
    mockup_rule = (
        "If the image is not a diagram but a user interface (a screenshot, mockup, "
        "wireframe or sketch of a web page or app screen), return exactly "
        f"{{\"diagramType\": \"{mockup_type}\"}} and nothing else.\n\n"
        if detect_mockup else ""
    )
    subject = "this image" if detect_mockup else "this image of a UML diagram"
    return (
        f"You are a UML/modeling expert. Analyze {subject}.\n\n"
        f"{mockup_rule}"
        "First, determine what TYPE of diagram it is, then extract its contents "
        "into the matching JSON format.\n\n"
        f"Supported formats:\n{ALL_SCHEMAS}\n\n"
        "IMPORTANT:\n"
        "1. Include the \"diagramType\" field matching the diagram you see\n"
        "2. Extract EVERY element visible in the diagram\n"
        "3. Keep names exactly as shown\n"
        "4. Return ONLY valid JSON, no explanations\n"
    )


def _get_schema_for_type(diagram_type: str) -> str:
    """Return the JSON schema string for a given diagram type."""
    return {
        "ClassDiagram": CLASS_DIAGRAM_SCHEMA,
        "StateMachineDiagram": STATE_MACHINE_SCHEMA,
        "ObjectDiagram": OBJECT_DIAGRAM_SCHEMA,
        "AgentDiagram": AGENT_DIAGRAM_SCHEMA,
        "BPMN": BPMN_SCHEMA,
    }.get(diagram_type, CLASS_DIAGRAM_SCHEMA)


def diagram_type_label(diagram_type: str) -> str:
    """Return a human-readable label for a diagram type (also used in the conversion reply)."""
    return {
        "ClassDiagram": "UML Class Diagram",
        "StateMachineDiagram": "UML State Machine Diagram",
        "ObjectDiagram": "UML Object Diagram",
        "AgentDiagram": "Agent / Chatbot Diagram",
        "BPMN": "BPMN Process Diagram",
    }.get(diagram_type, "UML Diagram")
