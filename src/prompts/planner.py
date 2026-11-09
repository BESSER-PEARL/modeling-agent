"""Request planner: the LLM prompt that splits a request into model / generation operations."""


FEW_SHOT_EXAMPLES = """
EXAMPLES:
User: "create a library system with books and authors" → [{"type":"model","diagramType":"ClassDiagram","mode":"complete_system","request":"create a library system with books and authors"}]
User: "create a User class" → [{"type":"model","diagramType":"ClassDiagram","mode":"modify_model","request":"create a User class"}]
User: "create a library system and generate django" → [{"type":"model","diagramType":"ClassDiagram","mode":"complete_system","request":"create a library system"},{"type":"generation","generatorType":"django"}]
User: "create a web app for a hotel booking" → [{"type":"model","diagramType":"ClassDiagram","mode":"complete_system","request":"create hotel booking system"},{"type":"model","diagramType":"GUINoCodeDiagram","mode":"complete_system","request":"create GUI for hotel booking"},{"type":"generation","generatorType":"web_app"}]
User: "add an email attribute to the User class" → [{"type":"model","diagramType":"ClassDiagram","mode":"modify_model","request":"add email attribute to User class"}]
User: "create a state machine for order processing" → [{"type":"model","diagramType":"StateMachineDiagram","mode":"complete_system","request":"create order processing state machine"}]
User: "generate python code" → [{"type":"generation","generatorType":"python"}]
User: "create a pizza ordering chatbot agent" → [{"type":"model","diagramType":"AgentDiagram","mode":"complete_system","request":"create pizza ordering chatbot agent"}]
User: "design an e-commerce system, create a gui for it, and generate a web app" → [{"type":"model","diagramType":"ClassDiagram","mode":"complete_system","request":"design e-commerce system"},{"type":"model","diagramType":"GUINoCodeDiagram","mode":"complete_system","request":"create GUI for e-commerce"},{"type":"generation","generatorType":"web_app"}]
User: "create a quantum circuit with 3 qubits and hadamard gates" → [{"type":"model","diagramType":"QuantumCircuitDiagram","mode":"complete_system","request":"create quantum circuit with 3 qubits and hadamard gates"}]
User: "add a constraint that a Library always has at least one Book" → [{"type":"model","diagramType":"ClassDiagram","mode":"modify_model","request":"add a constraint that a Library always has at least one Book"}]
User: "the precondition of Account::deposit is amount > 0" → [{"type":"model","diagramType":"ClassDiagram","mode":"modify_model","request":"the precondition of Account::deposit is amount > 0"}]
""".strip()


NON_WEB_APP_HINT = (
    "\nThe user wants an application that is NOT a web app. Do NOT plan "
    "GUINoCodeDiagram or web_app (the GUI editor designs web pages): plan "
    "the ClassDiagram; the code is generated afterwards from it."
)


def planner_prompt(message: str, context_summary: str, intent_hint: str, detected_targets_hint: str, generation_hint: str, generator_prerequisites: str) -> str:
    return f"""You are an assistant operation planner for BESSER modeling.

User request:
{message}

Workspace context:
{context_summary}{intent_hint}

Detected diagram targets (from user message keywords): {detected_targets_hint}{generation_hint}

Create a JSON plan with an "operations" array.
Operation types:
1) model:
{{
  "type": "model",
  "diagramType": "ClassDiagram|ObjectDiagram|StateMachineDiagram|AgentDiagram|GUINoCodeDiagram|QuantumCircuitDiagram|BPMN|ComponentDiagram|DeploymentDiagram",
  "mode": "complete_system|modify_model",
  "request": "sub-request focused ONLY on this specific diagram type"
}}
2) generation:
{{
  "type": "generation",
  "generatorType": "django|backend|web_app|sql|sqlalchemy|python|java|pydantic|jsonschema|smartdata|agent|qiskit",
  "config": {{}}
}}

Generator prerequisites (the planner must ensure these diagrams are created BEFORE the generation step):
{generator_prerequisites}

Rules:
- ONLY emit operations for diagram types the user EXPLICITLY asks for. Do NOT infer extra diagram types.
  For example, "Add a Payment class with status" → ONLY one ClassDiagram modify_model operation.
- Emit one model operation for EACH explicitly requested diagram target, in dependency order.
- ClassDiagram always comes first — other diagrams depend on it.
- Each model operation's "request" should be a focused sub-request for that specific diagram only.
  IMPORTANT: Each sub-request MUST contain enough detail from the original request so the
  handler knows what to generate. Don't just say "create a class diagram" — include the
  domain details like "create a class diagram for a library system with books, authors, and members".
- If the user asks for generation, emit a generation operation AFTER all required model operations.
- If prerequisite diagrams are missing from the workspace, create them first.
- For complex multi-step requests like "create X system then generate Y", decompose into:
  1. model operation (create the diagram)
  2. generation operation (generate the code)
- If the user says "create a web app for X", this means:
  1. Create ClassDiagram for X (complete_system)
  2. Create GUINoCodeDiagram for X (complete_system)
  3. Generate web_app code
- Keep operations minimal and deterministic.
- Return ONLY valid JSON: {{"operations":[...]}}.

{FEW_SHOT_EXAMPLES}
"""
