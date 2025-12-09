# 🤖 UML Modeling Agent

> **AI-powered conversational agent using BESSER Agentic framework for creating and modifying UML diagrams using natural language for the editor.besser-pearl.org**

The  Modeling Agent is an intelligent backend service that interprets natural language requests and generates UML diagram elements in real-time. It uses LLM (Large Language Models) to understand user intent and produce structured diagram specifications in the BESSER model format.

---

## 📋 Table of Contents

- [What the Bot Can Do](#-what-the-bot-can-do)
- [Supported Diagram Types](#-supported-diagram-types)
- [Architecture](#-architecture)
- [API & Data Formats](#-api--data-formats)
- [Installation & Setup](#-installation--setup)
- [Usage Examples](#-usage-examples)
- [Testing](#-testing)
- [Known Issues & Limitations](#-known-issues--limitations)

---

## ✅ What the Bot Can Do

### Core Capabilities

#### 1. **Class Diagrams** (Fully Supported ✅)
- ✅ Create individual classes with attributes and methods
- ✅ Generate complete class diagrams with multiple classes and relationships
- ✅ Support relationships: Association, Composition, Aggregation, Inheritance, Realization
- ✅ Specify visibility modifiers (+, -, #, ~)
- ✅ Define attribute types and method signatures
- ✅ Set multiplicity on relationships
- ✅ Modify existing classes

#### 2. **Object Diagrams** (Fully Supported ✅)
- ✅ Create object instances from class definitions
- ✅ **Reference class diagrams** to use exact class/attribute IDs
- ✅ Populate objects with realistic example values
- ✅ Create links between object instances
- ✅ Validate objects against their class definitions

#### 3. **State Machine Diagrams** (Fully Supported ✅)
- ✅ Create states (simple, initial, final, choice)
- ✅ Define transitions with triggers, guards, and effects
- ✅ Generate complete state machines with multiple states
- ✅ Support composite states and history states

#### 4. **Agent Diagrams** (Fully Supported ✅)
- ✅ Create agent nodes with goals and actions
- ✅ Define message passing between agents
- ✅ Generate multi-agent systems
- ✅ Specify agent communication protocols

#### 5. **UML Specification Queries** (Fully Supported ✅)
- ✅ Query official UML specification documents
- ✅ Get definitions of UML concepts and notation
- ✅ Retrieve best practices and formal definitions
- ✅ RAG-powered retrieval from UML specification PDFs

### Interaction Modes

- 🎯 **Single Element Creation**: "add a class User"
- 🎨 **Complete System Generation**: "create a library management system"
- 🔄 **Model Modification**: "add a method to the User class"
- 💬 **Natural Language Understanding**: Works with conversational requests

### Advanced Features

- 📊 **Context-Aware Generation**: Uses current diagram state to make intelligent decisions
- 🔗 **Reference Diagram Support**: ObjectDiagram can reference ClassDiagram definitions
- 🧠 **LLM-Powered**: Uses GPT for intelligent interpretation of user requests
- ⚡ **Real-Time WebSocket Communication**: Instant updates to frontend
- 🛡️ **Fallback Mechanisms**: Generates basic elements if AI fails


---

## 📊 Supported Diagram Types

| Diagram Type | Single Element | Complete System | Modify | Status |
|--------------|----------------|-----------------|--------|--------|
| **Class Diagram** | ✅ | ✅ | ✅ | Fully Supported |
| **Object Diagram** | ✅ | ✅ | ❌ | Fully Supported |
| **State Machine** | ✅ | ✅ | ❌ | Fully Supported |
| **Agent Diagram** | ✅ | ✅ | ❌ | Fully Supported |
| **UML Specification** | N/A | N/A | N/A | Fully Supported |

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │ (TypeScript React Widget)
│   (BESSER Web)  │
└────────┬────────┘
         │ WebSocket (JSON)
         ▼
┌─────────────────┐
│  Modeling Agent │ (Python Backend)
│  (BESSER AI)    │
├─────────────────┤
│ • Intent Router │
│ • Diagram       │
│   Handlers      │
│ • LLM Service   │
│ • RAG Service   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GPT/LLM       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   UML Specs     │
│   (Vector DB)   │
└─────────────────┘
```

### Key Components

1. **Intent Router** (`modeling_agent.py`):
   - Detects user intent (create, modify, query)
   - Routes requests to appropriate handlers
   - Manages conversation state

2. **Diagram Handlers** (`diagram_handlers/`):
   - `ClassDiagramHandler`: Handles class diagram generation
   - `ObjectDiagramHandler`: Handles object diagram generation (with reference support)
   - `StateMachineHandler`: Handles state machine generation
   - `AgentDiagramHandler`: Handles agent diagram generation

3. **LLM Service**:
   - Wraps GPT API calls
   - Handles prompt engineering
   - Parses and validates JSON responses

4. **RAG Service**:
   - Retrieval-Augmented Generation for UML specifications
   - Vector-based document retrieval from UML spec PDFs
   - Context-aware answers to UML specification questions
   - Leverages Chroma vector store for efficient document search

5. **WebSocket Service**:
   - Real-time communication with frontend
   - Sends structured BESSER model updates

---

## 📡 API & Data Formats

### Input Format (WebSocket JSON)

```json
{
  "message": "[DIAGRAM_TYPE:ClassDiagram] create a User class",
  "diagramType": "ClassDiagram",
  "currentModel": {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "elements": {},
    "relationships": {}
  },
  "referenceDiagramData": {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "elements": {...}
  }
}
```

### Output Format (BESSER Model)

#### Single Element Response

```json
{
  "action": "inject_element",
  "diagramType": "ClassDiagram",
  "element": {
    "name": "User",
    "attributes": [
      {"name": "id", "type": "String", "visibility": "+"},
      {"name": "name", "type": "String", "visibility": "+"}
    ],
    "methods": [
      {"name": "login", "returnType": "void", "visibility": "+"}
    ]
  },
  "message": "✅ Successfully created class User with 2 attributes and 1 method!"
}
```

#### Complete System Response

```json
{
  "action": "inject_complete_system",
  "diagramType": "ClassDiagram",
  "systemSpec": {
    "systemName": "LibraryManagement",
    "classes": [...],
    "relationships": [...]
  },
  "message": "✨ Created LibraryManagement diagram!"
}
```

### Object Diagram with Class Reference

```json
{
  "action": "inject_element",
  "diagramType": "ObjectDiagram",
  "element": {
    "objectName": "harryPotter",
    "className": "Book",
    "classId": "class_j14u331vy_mh3ca9ma",
    "attributes": [
      {
        "name": "title",
        "attributeId": "attr_gsral672h_mh3ca9ma",
        "value": "Harry Potter and the Sorcerer's Stone"
      },
      {
        "name": "isbn",
        "attributeId": "attr_zmzmcz7ud_mh3ca9ma",
        "value": "978-0439708180"
      }
    ]
  }
}
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Clone the repository
cd ModelingAgent

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config.ini.example config.ini
# Edit config.ini with your OpenAI API key
```

### Configuration (`config.ini`)

```ini
[DEFAULT]
API_KEY = your_openai_api_key_here
MODEL = gpt-4
TEMPERATURE = 0.7
PORT = 8765
```

### Running the Agent

```bash
python modeling_agent.py
```

The agent will start a WebSocket server on `ws://localhost:8765`

---

## 💡 Usage Examples

### Class Diagram Examples

```
User: "create a User class with id, name, and email attributes"
Agent: ✅ Created User class with 3 attributes

User: "add a login method to User"
Agent: ✅ Added login() method to User class

User: "create a library management system"
Agent: ✨ Created complete library system with Book, Author, Member, and Loan classes
```

### Object Diagram Examples

```
User: "create a book object named harryPotter"
Agent: ✅ Created harryPotter object (instance of Book) with 4 attributes

User: "add another book called lordOfTheRings"
Agent: ✅ Created lordOfTheRings object with realistic values
```

### State Machine Examples

```
User: "create a login state machine"
Agent: ✨ Created login state machine with Idle, Authenticating, and LoggedIn states

User: "add a timeout transition"
Agent: ✅ Added timeout transition from Authenticating to Idle
```

### Agent Diagram Examples

```
User: "create a multi-agent auction system"
Agent: ✨ Created auction system with Auctioneer and Bidder agents
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_class_diagram_handler.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=diagram_handlers
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                     # Shared fixtures
├── test_class_diagram_handler.py   # Class diagram tests
├── test_payload_and_routing.py     # Integration tests
└── test_workflow_scenarios.py      # End-to-end tests
```
---


---

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is part of the BESSER framework. See LICENSE for details.

---
