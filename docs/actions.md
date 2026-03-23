# Actions Reference

All actions used in the WebSocket protocol between the backend (modeling agent) and the frontend (web modeling editor).

---

## Top-Level Actions

### Backend → Frontend

| Action | Purpose | Source Files |
|---|---|---|
| `inject_element` | Add a single element to the diagram canvas | All diagram handlers |
| `inject_complete_system` | Inject a full diagram (all elements + relationships) | All diagram handlers, `file_conversion_handler.py` |
| `modify_model` | Apply modifications to existing diagram elements | All diagram handlers, `base_handler.py` |
| `assistant_message` | Text-only message displayed in the chat panel | `session_helpers.py`, `execution.py`, `generation_handler.py` |
| `stream_start` | Begin a streaming text response | `session_helpers.py` |
| `stream_chunk` | One token/chunk of a streaming text response | `session_helpers.py` |
| `stream_done` | End of a streaming text response | `session_helpers.py` |
| `progress` | Loading/progress indicator update | `session_helpers.py` |
| `agent_error` | Error payload sent when something goes wrong | `execution.py`, `state_bodies.py`, `file_conversion_handler.py` |
| `switch_diagram` | Switch the active diagram tab in the editor | `workspace_orchestrator.py` |
| `trigger_generator` | Trigger code generation from the current model | `generation_handler.py` |
| `trigger_export` | Trigger model export (e.g., to file) | `generation_handler.py` |
| `trigger_deploy` | Trigger deployment of the generated application | `generation_handler.py` |
| `auto_generate_gui` | Auto-generate a GUI diagram from a class diagram | `confirmation.py` |

### Frontend → Backend

| Action | Purpose | Source Files |
|---|---|---|
| `user_message` | Inbound message from the user/frontend | `request_builders.py`, `README.md` |

---

## Nested Modification Actions

These actions appear **inside** the `modification` or `modifications` array within a `modify_model` top-level action.

### Class Diagram

| Action | Purpose | Source |
|---|---|---|
| `modify_class` | Modify an existing class (rename, change visibility, etc.) | `class_diagram_handler.py` |
| `add_class` | Add a new class to the diagram | `class_diagram_handler.py` |
| `remove_element` | Remove a class or relationship from the diagram | `class_diagram_handler.py` |
| `add_relationship` | Add a new relationship between classes | `class_diagram_handler.py` |
| `add_attribute` | Add an attribute to a class | `class_diagram_handler.py` |
| `remove_method` | Remove a method from a class | `class_diagram_handler.py` |
| `modify_attribute` | Modify an existing attribute on a class | `class_diagram_handler.py` |

### Object Diagram

| Action | Purpose | Source |
|---|---|---|
| `modify_object` | Modify an existing object instance | `object_diagram_handler.py` |

### State Machine Diagram

| Action | Purpose | Source |
|---|---|---|
| `modify_state` | Modify an existing state | `state_machine_handler.py` |

### Agent Diagram

| Action | Purpose | Source |
|---|---|---|
| `modify_state` | Modify a state in the agent diagram | `agent_diagram_handler.py` |

### Generic (Base Handler)

| Action | Purpose | Source |
|---|---|---|
| `modify_element` | Generic element modification (fallback) | `base_handler.py` |

---

## Payload Examples

### `inject_element`

```json
{
  "action": "inject_element",
  "diagramType": "ClassDiagram",
  "diagramId": "diagram-001",
  "element": {
    "className": "User",
    "attributes": [
      {"name": "id", "type": "String", "visibility": "public"}
    ],
    "methods": [],
    "position": {"x": 100, "y": 200}
  },
  "message": "Created the **User** class.",
  "suggestedActions": [
    {"label": "Add Order class", "prompt": "Add an Order class"}
  ]
}
```

### `inject_complete_system`

```json
{
  "action": "inject_complete_system",
  "diagramType": "ClassDiagram",
  "diagramId": "diagram-001",
  "systemSpec": {
    "systemName": "E-commerce System",
    "classes": [
      {
        "className": "User",
        "attributes": [
          {"name": "id", "type": "String", "visibility": "public"},
          {"name": "email", "type": "String", "visibility": "public"}
        ],
        "methods": [],
        "position": {"x": 100, "y": 200}
      },
      {
        "className": "Order",
        "attributes": [
          {"name": "orderId", "type": "String", "visibility": "public"}
        ],
        "methods": [],
        "position": {"x": 350, "y": 200}
      }
    ],
    "relationships": [
      {
        "type": "Association",
        "source": "User",
        "target": "Order",
        "sourceMultiplicity": "1",
        "targetMultiplicity": "0..*",
        "name": "creates"
      }
    ]
  },
  "replaceExisting": false,
  "createNewTab": false,
  "message": "Built the **E-commerce System** class diagram with 2 classes and 1 relationship.",
  "suggestedActions": [
    {"label": "Add Product class", "prompt": "Add a Product class"}
  ]
}
```

### `modify_model`

```json
{
  "action": "modify_model",
  "diagramType": "ClassDiagram",
  "diagramId": "diagram-001",
  "modification": {
    "action": "modify_class",
    "target": {"className": "User"},
    "changes": {"name": "Customer"}
  },
  "message": "Renamed **User** to **Customer**."
}
```

### `modify_model` (batch)

```json
{
  "action": "modify_model",
  "diagramType": "ClassDiagram",
  "diagramId": "diagram-001",
  "modifications": [
    {
      "action": "add_attribute",
      "target": {"className": "User"},
      "changes": {"name": "phone", "type": "String", "visibility": "public"}
    },
    {
      "action": "remove_method",
      "target": {"className": "Order"},
      "changes": {"name": "deprecatedMethod"}
    }
  ],
  "message": "Added **phone** to User and removed **deprecatedMethod** from Order."
}
```

### `assistant_message`

```json
{
  "action": "assistant_message",
  "message": "The User class represents the authenticated users in the system."
}
```

### `stream_start` / `stream_chunk` / `stream_done`

```json
{"action": "stream_start"}
{"action": "stream_chunk", "chunk": "The class diagram "}
{"action": "stream_chunk", "chunk": "represents the core "}
{"action": "stream_chunk", "chunk": "domain model."}
{"action": "stream_done"}
```

### `progress`

```json
{
  "action": "progress",
  "message": "Generating class diagram..."
}
```

### `agent_error`

```json
{
  "action": "agent_error",
  "message": "Failed to generate the diagram.",
  "errorCode": "LLM_PARSE_ERROR",
  "retryable": true
}
```

### `switch_diagram`

```json
{
  "action": "switch_diagram",
  "diagramType": "StateMachineDiagram",
  "diagramId": "diagram-002"
}
```

### `trigger_generator`

```json
{
  "action": "trigger_generator",
  "generator": "python",
  "message": "Generating Python code from your model..."
}
```

### `trigger_export`

```json
{
  "action": "trigger_export",
  "format": "json",
  "message": "Exporting your model..."
}
```

### `trigger_deploy`

```json
{
  "action": "trigger_deploy",
  "message": "Deploying your application..."
}
```

### `auto_generate_gui`

```json
{
  "action": "auto_generate_gui",
  "sourceDigramType": "ClassDiagram",
  "message": "Auto-generating GUI from the class diagram..."
}
```
