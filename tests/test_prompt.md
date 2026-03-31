# Modeling Agent — Manual Test Plan

## Class Diagram — Core

| # | Prompt | Expected Result |
|---|--------|-----------------|
| 1 | "Create a library management system" | 4-6 classes with attributes, relationships, multiplicities. No code on methods (`implementationType=none`) |
I also want to store information about the users of the library and the books they read
| 2 | "Add a Payment class with amount, date, and status" | Payment class added. Then "link it to Library" adds a relationship |
| 3 | "Create an abstract class Vehicle with speed and color" | `type: AbstractClass`, italic name, `<<abstract>>` stereotype |
| 4 | "Create an enumeration OrderStatus with PENDING, PROCESSING, SHIPPED, DELIVERED and add it as an attribute of the book too" | `type: Enumeration` with values (no `: str`). Book gets `orderStatus: OrderStatus` |
| 5 | "Add a derived attribute totalPrice to Book" | Attribute shows with `/` prefix |
| 6 | "Add an optional attribute middleName to Author" | Attribute shows with `?` suffix |
| 7 | "Set default value of name of Author to 'john doe'" | Attribute shows `= john doe` |
| 8 | "Rename User to Customer" | `modify_class` works on any class type (Class, AbstractClass, Enumeration) |

## Class Diagram — Code Generation

| # | Prompt | Expected Result |
|---|--------|-----------------|
| 9 | "Create a Calculator class with add and subtract methods, implement them in Python" | Methods have `implementationType: code` with Python in the `code` field |
| 10 | "Implement the methods in BAL" | `implementationType: bal` with BAL syntax (`def name() -> type { ... }`) |

## Class Diagram — Confirmation Flow

| # | Prompt | Expected Result |
|---|--------|-----------------|
| 11 | With existing model: "Create an e-commerce system" | Asks replace / keep / new tab |
| 12 | Upload a PDF (with existing model) | Asks replace / keep / new tab |
| 13 | Via voice: "create a school system" (with existing model) | Asks replace / keep / new tab |

## State Machine

| # | Prompt | Expected Result |
|---|--------|-----------------|
| 14 | "Create a traffic light state machine" | Complete system with states + transitions |
| 15 | "Add a Processing state with entry action startProcessing" | `add_state` with `StateBody` for entry action |
| 16 | "Add an initial state" | Creates `StateInitialNode` |
| 17 | "Add a final state" | Creates `StateFinalNode` |
| 18 | "Add a code block with a function that validates input" | Creates `StateCodeBlock` element |
| 19 | "Add a transition from Idle to Processing with trigger startProcess" | `add_transition` between states |

## Object Diagram

| # | Prompt | Expected Result |
|---|--------|-----------------|
| 20 | "Create objects for a library system" (with class diagram as reference) | Objects with concrete attribute values |
| 21 | "Add an object user2 of class User with name Bob" | `add_object` with attribute values |

## Agent Diagram

| # | Prompt | Expected Result |
|---|--------|-----------------|
| 22 | "Create a FAQ chatbot agent" | States, intents, and transitions created |
| 23 | "Add a welcome state with reply 'Hello! How can I help?'" | `add_state` with `replyType: text` |
| 24 | "Add a greeting intent with phrases hello, hi, hey" | `add_intent` with training phrases |
| 25 | "Add a state with an LLM reply" | `replyType: llm` |
| 26 | "Add a RAG knowledge base called ProductDocs" | Creates `AgentRagElement` (cylinder shape) |
| 27 | "Add a state that queries the database" | `replyType: db_reply` |

## Templates

| # | Action | Expected Result |
|---|--------|-----------------|
| 28 | Open template library | Shows Behavioral (Command, Observer), Creational (Factory), GUI categories |
| 29 | Load the Command pattern template | Creates a class diagram |

## Edge Cases

| # | Scenario | Expected Result |
|---|----------|-----------------|
| 30 | Empty canvas + "create a class User" | Works via `modify_model` (no `single_element` mode) |
| 31 | Send a message via voice | Agent sees full canvas context |
| 32 | Upload a PlantUML file | Converts and injects (or asks confirmation if model exists) |
| 33 | Upload a PDF file | Converts and injects (or asks confirmation if model exists) |
| 34 | Create enum, then "add a value to it" | Adds to existing enum, does NOT re-create it |
| 35 | Two quick follow-up messages | Agent uses conversation memory, no duplicate operations |
