# Intelligent UML Modeling Assistant agent
# Supports: ClassDiagram, ObjectDiagram, StateMachineDiagram, AgentDiagram

import logging
import json
import re
import random
import sys
import os
from typing import Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from besser.agent import nlp
from besser.agent.core.agent import Agent
from besser.agent.core.session import Session
from besser.agent.library.transition.events.base_events import ReceiveJSONEvent
from besser.agent.exceptions.logger import logger
from besser.agent.nlp.intent_classifier.intent_classifier_configuration import LLMIntentClassifierConfiguration
from besser.agent.nlp.llm.llm_openai_api import LLMOpenAI
from besser.agent.nlp.rag.rag import RAGMessage, RAG

from diagram_handlers.factory import DiagramHandlerFactory, get_diagram_type_info
from diagram_handlers.utils import (
    extract_diagram_type_from_message,
    detect_diagram_type_from_keywords
)

# Layout defaults for newly generated elements
LAYOUT_BASE_X = -940
LAYOUT_BASE_Y = -600
LAYOUT_X_SPREAD = 360
LAYOUT_Y_SPREAD = 280
CLASS_WIDTH = 220
ATTRIBUTE_HEIGHT = 25
METHOD_HEIGHT = 25
CLASS_HEADER_HEIGHT = 50

# Configure the logging module
logger.setLevel(logging.INFO)

# Create the agent
agent = Agent('uml_modeling_agent')

agent.load_properties('config.ini')
print("✅ Agent properties loaded from config.ini")
print(f" - Agent Name: {agent.name}")

websocket_platform = agent.use_websocket_platform(use_ui=False)


def extract_json_payload(session: Session) -> Dict[str, Any]:
    """Extract the JSON payload from the current session event.
    
    With unified JSON messages, the event contains:
    - message: The user's text message (used for intent classification)
    - diagramType: The diagram type context
    - currentModel: The full model context (optional)
    """
    if not session.event:
        return {}
    
    # For ReceiveJSONEvent, the payload is in event.json
    if isinstance(session.event, ReceiveJSONEvent):
        return session.event.json or {}
    
    # For text messages, try to parse as JSON
    if hasattr(session.event, 'message'):
        message = session.event.message
        if isinstance(message, str) and message.strip().startswith('{'):
            try:
                return json.loads(message)
            except Exception:
                pass
    
    return {}


def get_user_message(session: Session) -> str:
    """Extract the user's message from the session event.
    
    For JSON events, this extracts the 'message' field.
    For text events, this returns the message directly.
    """
    payload = extract_json_payload(session)
    
    # First check for message in JSON payload
    if payload and 'message' in payload:
        message = payload['message']
        # Clean up diagram type prefix if present
        if isinstance(message, str):
            prefix_match = re.match(r'^\[DIAGRAM_TYPE:\w+\]\s*(.+)', message, re.DOTALL)
            if prefix_match:
                return prefix_match.group(1).strip()
            return message.strip()
    
    # Fallback to event message
    if hasattr(session.event, 'message') and session.event.message:
        message = session.event.message
        if isinstance(message, str):
            # Check if it's a JSON string
            if message.strip().startswith('{'):
                try:
                    parsed = json.loads(message)
                    if isinstance(parsed, dict) and 'message' in parsed:
                        inner = parsed['message']
                        prefix_match = re.match(r'^\[DIAGRAM_TYPE:\w+\]\s*(.+)', inner, re.DOTALL)
                        if prefix_match:
                            return prefix_match.group(1).strip()
                        return inner.strip() if isinstance(inner, str) else str(inner)
                except Exception:
                    pass
            # Clean diagram type prefix
            prefix_match = re.match(r'^\[DIAGRAM_TYPE:\w+\]\s*(.+)', message, re.DOTALL)
            if prefix_match:
                return prefix_match.group(1).strip()
            return message.strip()
    
    return ""


def get_diagram_type(session: Session, default: str = 'ClassDiagram') -> str:
    """Extract the diagram type from the session event."""
    payload = extract_json_payload(session)
    
    # Check payload for diagramType
    if payload and payload.get('diagramType'):
        return payload['diagramType']
    
    # Check message for prefix
    message = get_user_message(session)
    if message:
        extracted = extract_diagram_type_from_message(message)
        if extracted:
            return extracted
        detected = detect_diagram_type_from_keywords(message)
        if detected:
            return detected
    
    return default


def get_current_model(session: Session) -> Optional[Dict[str, Any]]:
    """Extract the current model from the session event's payload."""
    payload = extract_json_payload(session)
    return payload.get('currentModel') if payload else None


# Intent matching condition functions for JSON events
def json_intent_matches(session: Session, params: Dict[str, Any]) -> bool:
    """Check if the predicted intent matches the target intent for JSON events."""
    target_intent_name = params.get('intent_name')
    if not target_intent_name:
        return False
    
    # The ReceiveJSONEvent should have predicted_intent after intent prediction
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == target_intent_name
    
    return False


def json_no_intent_matched(session: Session) -> bool:
    """Check if no specific intent was matched (fallback).
    
    Note: This function takes only session (no params) because it doesn't need any parameters.
    """
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == 'fallback_intent'
    return True


def generate_layout_position(seed: Optional[str] = None) -> Dict[str, int]:
    """Compute a deterministic layout position near the top-left workspace area."""
    rng = random.Random(seed) if seed else random
    x_offset = rng.randint(0, LAYOUT_X_SPREAD)
    y_offset = rng.randint(0, LAYOUT_Y_SPREAD)
    return {
        'x': LAYOUT_BASE_X + x_offset,
        'y': LAYOUT_BASE_Y + y_offset
    }

try:
    gpt = LLMOpenAI(
        agent=agent,
        name='gpt-4o-mini',
        parameters={
            'temperature': 0.4,
            'max_tokens': 3000
        },
        num_previous_messages=3
    )
    
    if gpt is None:
        raise Exception("LLM initialization returned None")
    
    logger.info("✅ LLM initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}")
    print("\n" + "="*80)
    print("ERROR: Failed to Initialize OpenAI LLM")
    print("="*80)
    print(f"\nError: {e}")
    print("\nPlease check:")
    print("1. Your OpenAI API key in config.ini (line: nlp.openai.api_key)")
    print("2. The key format should be: sk-proj-... or sk-...")
    print("3. The key has not expired or been revoked")
    print("4. You have credits available in your OpenAI account")
    print("\nGet your key from: https://platform.openai.com/api-keys")
    print("="*80 + "\n")
    exit(1)

gpt_complex = gpt

# Create Vector Store for UML Specification RAG
try:
    vector_store: Chroma = Chroma(
        embedding_function=OpenAIEmbeddings(openai_api_key=agent.get_property(nlp.OPENAI_API_KEY)),
        persist_directory='uml_vector_store'
    )
    # Create text splitter (RAG creates a vector for each chunk)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Create the RAG for UML specification queries
    uml_rag = RAG(
        agent=agent,
        vector_store=vector_store,
        splitter=splitter,
        llm_name='gpt-4o-mini',
        k=4,
        num_previous_messages=0
    )
    
    uml_rag.llm_prompt = """You are a UML (Unified Modeling Language) specification expert. Based on the context retrieved from the UML specification documents, answer the user's question about UML concepts, notation, semantics, or best practices.

If the context contains relevant information, use it to provide an accurate and detailed answer.
If you don't find the answer in the context, say that you don't have that specific information in the UML specification documents, but you can provide general guidance based on your knowledge.

Be precise and reference specific UML concepts when applicable. Use clear examples when helpful."""
    
    # Uncomment the following line to load UML specification PDFs into the vector store
    uml_rag.load_pdfs('./uml_specs')
    
    logger.info("✅ UML RAG initialized successfully")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize UML RAG: {e}. RAG features will be disabled.")
    uml_rag = None

# Initialize diagram handler factory
diagram_factory = DiagramHandlerFactory(gpt)
logger.info(f"✅ Diagram handlers initialized: {', '.join(diagram_factory.get_supported_types())}")

ic_config = LLMIntentClassifierConfiguration(
    llm_name='gpt-4o-mini',
    parameters={},
    use_intent_descriptions=True,
    use_training_sentences=False,
    use_entity_descriptions=True,
    use_entity_synonyms=False
)

agent.set_default_ic_config(ic_config)

# STATES
greetings_state = agent.new_state('greetings_state', initial=True)
create_single_element_state = agent.new_state('create_single_element_state')
create_complete_system_state = agent.new_state('create_complete_system_state')
modify_model_state = agent.new_state('modify_model_state')
modeling_help_state = agent.new_state('modeling_help_state')
clarify_diagram_type_state = agent.new_state('clarify_diagram_type_state')
uml_rag_state = agent.new_state('uml_rag_state')

# INTENTS
hello_intent = agent.new_intent(
    name='hello_intent',
    description='The user greets you or wants to start a conversation'
)

create_single_element_intent = agent.new_intent(
    name='create_single_element_intent',
    description='The user wants to create exactly ONE single UML element. Examples: "create a class called User", "add a Person class", "make one state", "create an object instance". This is for creating ONE element only, NOT multiple elements or systems.'
)

create_complete_system_intent = agent.new_intent(
    name='create_complete_system_intent',
    description='The user wants to create a complete system, diagram, or multiple classes/elements. Keywords: "create a library system", "create a class diagram for", "design an e-commerce", "generate a banking application", "build a system", "create a diagram for", "model a", "create classes for". This is for creating MULTIPLE elements or a complete model.'
)

modify_model_intent = agent.new_intent(
    name='modify_model_intent',
    description='The user wants to modify, change, update, edit, add to, or connect elements in an EXISTING UML model. Keywords: "add relationship", "connect", "add inheritance", "add generalization", "relate", "modify class", "change attribute", "update method", "delete", "remove", "rename", "add association", "add composition", "add aggregation", "link classes"'
)

modeling_help_intent = agent.new_intent(
    name='modeling_help_intent',
    description='The user asks for help with UML modeling, design patterns, or modeling concepts'
)

uml_spec_intent = agent.new_intent(
    name='uml_spec_intent',
    description='The user asks theoretical questions about the official UML specification document, UML standards, or formal UML definitions. Keywords: "according to UML specification", "what does UML standard say", "UML 2.5 specification", "OMG specification", "formal UML definition". This is NOT for creating diagrams, only for asking about the UML specification document itself.'
)

# STATE BODY DEFINITIONS

def generate_model_modification(user_request: str, current_model: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate model modifications based on user request and current model context"""
    system_prompt = """You are a UML modeling expert. The user wants to modify an existing UML model.

Return ONLY a JSON object with this structure:
{
  "action": "modify_model",
  "modification": {
    "action": "modify_class" | "modify_attribute" | "modify_method" | "add_relationship" | "remove_element",
    "target": {
      "classId": "optional_id",
      "className": "ClassName",
      "attributeId": "optional_attr_id",
      "attributeName": "originalAttributeName",
      "methodId": "optional_method_id",
      "methodName": "originalMethodName",
      "relationshipId": "optional_relationship_id",
      "relationshipName": "existingRelationshipName",
      "sourceClass": "SourceClass",
      "targetClass": "TargetClass"
    },
    "changes": {
      "name": "newName",
      "type": "newType",
      "visibility": "public|private|protected",
      "parameters": [{"name": "param", "type": "String"}],
      "returnType": "ReturnType",
      "sourceMultiplicity": "1",
      "targetMultiplicity": "*",
      "relationshipType": "Association|Aggregation|Composition|Inheritance"
    }
  },
  "message": "Short explanation of what changed"
}

Rules:
1. Only reference classes or elements that exist in the provided model context.
2. When renaming attributes or methods include the ORIGINAL name in target.attributeName/target.methodName.
3. When adding relationships include sourceClass, targetClass, relationshipType, and multiplicities.
4. Prefer ids if provided in the context.
5. Keep the message user-friendly and concise.
6. Do not invent new classes unless the user explicitly asks to add them."""

    context_info = []
    if current_model and isinstance(current_model, dict) and current_model.get('elements'):
        elements = current_model.get('elements', {})
        for element in elements.values():
            if element.get('type') == 'Class':
                name = element.get('name', 'Unknown')
                attr_names = []
                for attr_id in element.get('attributes', []) or []:
                    attr = elements.get(attr_id, {})
                    attr_name = attr.get('name')
                    if attr_name:
                        attr_names.append(attr_name)
                method_names = []
                for method_id in element.get('methods', []) or []:
                    method = elements.get(method_id, {})
                    method_name = method.get('name')
                    if method_name:
                        method_names.append(method_name)
                summary = f"Class {name}"
                if attr_names:
                    summary += f" | attributes: {', '.join(attr_names[:4])}"
                if method_names:
                    summary += f" | methods: {', '.join(method_names[:4])}"
                context_info.append(summary)

    context_block = ''
    if context_info:
        context_block = "\n\nCurrent model summary:\n- " + "\n- ".join(context_info[:8])

    user_prompt = f"Modify the UML model: {user_request}{context_block}"

    try:
        response = gpt.predict(f"{system_prompt}\n\nUser Request: {user_prompt}")

        json_text = response.strip()
        if json_text.startswith('```json'):
            json_text = json_text[7:]
        if json_text.endswith('```'):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        modification_spec = json.loads(json_text)
        if not isinstance(modification_spec, dict):
            raise ValueError("Modification response is not a JSON object")

        modification_spec.setdefault('action', 'modify_model')
        if 'modification' not in modification_spec:
            raise ValueError("Missing modification payload")

        modification_spec.setdefault(
            'message',
            f"Applied {modification_spec['modification'].get('action', 'modification')} to {modification_spec['modification'].get('target', {}).get('className', 'model')}"
        )
        
        # logger.info(f"[LLM] Final modification spec being sent:")
        # logger.info(f"  - Action: {modification_spec['modification'].get('action')}")
        # logger.info(f"  - Target: {modification_spec['modification'].get('target')}")
        # logger.info(f"  - Changes: {modification_spec['modification'].get('changes')}")

        return modification_spec

    except Exception as e:
        logger.error(f"Error generating model modification: {e}")
        return {
            'action': 'modify_model',
            'modification': {
                'action': 'modify_class',
                'target': {'className': 'Unknown'},
                'changes': {'name': 'ModifiedClass'}
            },
            'message': 'Failed to generate modification automatically (fallback used).'
        }


# STATE BODY DEFINITIONS

def global_fallback_body(session: Session):
    """Handle unrecognized messages."""
    user_message = get_user_message(session) or "your message"
    answer = gpt.predict(f"You are a UML modeling assistant. The user said: '{user_message}'. If this is related to UML modeling, suggest how you can help them create models, classes, or diagrams. Otherwise, politely explain that you specialize in UML modeling assistance.")
    session.reply(answer)

agent.set_global_fallback_body(global_fallback_body)

def greetings_body(session: Session):
    """Send a greeting message when the user first connects or says hello."""
    greeting_message = """Hello! I'm your UML Assistant!

I can help you:
- Create classes: "Create a User class"
- Build systems: "Create a library management system"
- Create agent diagrams: "Create an agent"
- Modify diagrams: "Add transition from welcome to menu"
- UML specification: "What does UML say about association classes?"

What would you like to create?"""

    # On initial state entry, session.event is None and connection isn't ready yet
    # Wait for the frontend's "hello" message to trigger the greeting
    if session.event is None:
        return
    
    # Check if this is a hello intent
    is_hello_intent = False
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        is_hello_intent = session.event.predicted_intent.intent.name == 'hello_intent'
    
    # If user said hello and we haven't greeted yet, send full greeting
    if is_hello_intent and not session.get('has_greeted'):
        session.reply(greeting_message)
        session.set('has_greeted', True)
        return
    
    # If user said hello again after initial greeting, send short response
    if is_hello_intent and session.get('has_greeted'):
        session.reply("Hello again! How can I help you with UML modeling?")
        return


greetings_state.set_body(greetings_body)

# Transitions from greetings state
# Support both text events (backward compatibility) and JSON events (unified messages)
# Text event transitions (for backward compatibility)
greetings_state.when_intent_matched(hello_intent).go_to(greetings_state)
greetings_state.when_intent_matched(create_single_element_intent).go_to(create_single_element_state)
greetings_state.when_intent_matched(create_complete_system_intent).go_to(create_complete_system_state)
greetings_state.when_intent_matched(modify_model_intent).go_to(modify_model_state)
greetings_state.when_intent_matched(modeling_help_intent).go_to(modeling_help_state)
greetings_state.when_intent_matched(uml_spec_intent).go_to(uml_rag_state)
# JSON event transitions (for unified messages)
# Note: ReceiveJSONEvent supports intent classification via predict_intent() when message field is present
greetings_state.when_event(ReceiveJSONEvent())\
    .with_condition(json_intent_matches, {'intent_name': 'hello_intent'})\
    .go_to(greetings_state)
greetings_state.when_event(ReceiveJSONEvent())\
    .with_condition(json_intent_matches, {'intent_name': 'create_single_element_intent'})\
    .go_to(create_single_element_state)
greetings_state.when_event(ReceiveJSONEvent())\
    .with_condition(json_intent_matches, {'intent_name': 'create_complete_system_intent'})\
    .go_to(create_complete_system_state)
greetings_state.when_event(ReceiveJSONEvent())\
    .with_condition(json_intent_matches, {'intent_name': 'modify_model_intent'})\
    .go_to(modify_model_state)
greetings_state.when_event(ReceiveJSONEvent())\
    .with_condition(json_intent_matches, {'intent_name': 'modeling_help_intent'})\
    .go_to(modeling_help_state)
greetings_state.when_event(ReceiveJSONEvent())\
    .with_condition(json_intent_matches, {'intent_name': 'uml_spec_intent'})\
    .go_to(uml_rag_state)
greetings_state.when_event(ReceiveJSONEvent())\
    .with_condition(json_no_intent_matched)\
    .go_to(modeling_help_state)
greetings_state.when_no_intent_matched().go_to(modeling_help_state)

def extract_modeling_context(session: Session) -> Optional[Dict[str, Any]]:
    """Normalize request data for the specialized modeling states.
    
    With unified JSON messages, the context is extracted directly from the JSON payload.
    """
    # Use the new helper functions for unified JSON handling
    actual_message = get_user_message(session)
    diagram_type = get_diagram_type(session)
    current_model = get_current_model(session)
    payload_data = extract_json_payload(session)
    
    if not actual_message:
        return None

    diagram_info = get_diagram_type_info(diagram_type)
    handler = diagram_factory.get_handler(diagram_type)
    
    # Log the diagram type and handler being used for this request
    logger.info(f"📊 {diagram_type} | Handler: {handler.__class__.__name__ if handler else 'None'}")
    
    return {
        'user_message': actual_message,
        'actual_message': actual_message,
        'diagram_type': diagram_type,
        'payload_data': payload_data,
        'current_model': current_model,
        'diagram_info': diagram_info,
        'handler': handler
    }

def create_single_element_body(session: Session):
    """Generate a single UML element based on the user's request.
    
    With unified JSON messages, this processes the request directly without
    waiting for a second message.
    """
    # Store which intent brought us here
    session.set('last_matched_intent', 'create_single_element_intent')
    
    context = extract_modeling_context(session)
    if not context:
        session.reply("I need more details about what you'd like to create. Could you describe it?")
        return

    handler = context['handler']
    diagram_type = context['diagram_type']
    actual_message = context['actual_message']
    current_model = context['current_model']

    if not handler:
        session.reply(f"Warning: {diagram_type} is not supported yet. Please use ClassDiagram for now.")
        return

    # Process the request directly (unified JSON contains all context)
    try:
        # Extract reference diagram if available (for ObjectDiagram)
        reference_diagram = None
        if current_model and isinstance(current_model, dict):
            reference_diagram = current_model.get('referenceDiagramData')
        
        # Call handler with reference diagram if it supports it (ObjectDiagram)
        if diagram_type == 'ObjectDiagram' and hasattr(handler, 'generate_single_element'):
            result = handler.generate_single_element(actual_message, reference_diagram=reference_diagram)
        else:
            result = handler.generate_single_element(actual_message)

        if result and result.get('element'):
            result['diagramType'] = diagram_type
            session.reply(json.dumps(result))
        else:
            session.reply("I had trouble creating that element. Could you provide more details?")

    except Exception as e:
        logger.error(f"Error in create_single_element_body: {e}")
        session.reply("I encountered an issue while creating the element. Could you try rephrasing your request?")

create_single_element_state.set_body(create_single_element_body)


def create_complete_system_body(session: Session):
    """Generate a complete system with multiple elements and relationships.
    
    With unified JSON messages, this processes the request directly without
    waiting for a second message.
    """
    # Store which intent brought us here
    session.set('last_matched_intent', 'create_complete_system_intent')
    
    context = extract_modeling_context(session)
    if not context:
        session.reply("I need more details about the system you'd like to create. Could you describe it?")
        return

    handler = context['handler']
    diagram_type = context['diagram_type']
    actual_message = context['actual_message']

    if not handler:
        session.reply(f"Warning: {diagram_type} is not supported yet. Please use ClassDiagram for now.")
        return

    # Process the request directly (unified JSON contains all context)
    try:
        result = handler.generate_complete_system(actual_message)

        if result and result.get('systemSpec'):
            result['diagramType'] = diagram_type
            session.reply(json.dumps(result))
        else:
            session.reply("I had trouble generating that system. Could you provide more details?")

    except Exception as e:
        logger.error(f"Error in create_complete_system_body: {e}")
        session.reply("I encountered an issue while creating the system. Could you try rephrasing your request?")

create_complete_system_state.set_body(create_complete_system_body)

def modify_modeling_body(session: Session):
    """Apply modifications to an existing UML model."""
    # Store which intent brought us here
    session.set('last_matched_intent', 'modify_model_intent')
    
    context = extract_modeling_context(session)
    if not context:
        return

    handler = context['handler']
    diagram_type = context['diagram_type']
    actual_message = context['actual_message']
    current_model = context['current_model']

    if not handler:
        session.reply(f"Warning: {diagram_type} is not supported yet. Please use ClassDiagram for now.")
        return

    # Process the modification request directly (unified JSON contains all context)
    try:
        # Use the handler's specialized generate_modification method
        modification_spec = handler.generate_modification(actual_message, current_model)

        if modification_spec and modification_spec.get('modification'):
            modification_spec['diagramType'] = diagram_type

            # # Log what we're sending
            # logger.info(f"[MODIFY] Sending modification to frontend:")
            # logger.info(f"  - Action: {modification_spec.get('action')}")
            # logger.info(f"  - Modification action: {modification_spec['modification'].get('action')}")
            # logger.info(f"  - Target: {modification_spec['modification'].get('target')}")
            # logger.info(f"  - Changes: {modification_spec['modification'].get('changes')}")

            session.reply(json.dumps(modification_spec))
        else:
            session.reply("I couldn't determine the modification to apply. Could you provide more detail?")

    except Exception as e:
        logger.error(f"Error in modify_modeling_body: {e}")
        session.reply("I encountered an issue while updating the model. Could you try rephrasing your request?")

modify_model_state.set_body(modify_modeling_body)

def modeling_help_body(session: Session):
    """Offer guidance or clarifying questions when the user needs modeling help.
    
    With unified JSON messages, this processes the request directly.
    """
    # Store which intent brought us here
    session.set('last_matched_intent', 'modeling_help_intent')
    
    context = extract_modeling_context(session)
    if not context:
        # Provide general help if no context
        session.reply("I can help you with UML modeling! Try asking me to create a class, design a system, or modify your diagram.")
        return

    diagram_info = context['diagram_info']
    actual_message = context['actual_message']

    # Process the help request directly (unified JSON contains all context)
    help_prompt = f"""You are a UML modeling expert assistant working with {diagram_info['name']}. The user asked: "{actual_message}"

Current diagram type: {diagram_info['name']} - {diagram_info['description']}

Provide helpful, practical advice about UML modeling for this diagram type. If they're asking about concepts, explain them clearly. If they want to create something, guide them on how to express their requirements.

Keep your response conversational and encouraging. Suggest specific things they can ask you to create."""

    try:
        answer = gpt.predict(help_prompt)
        session.reply(answer)
    except Exception as e:
        logger.error(f"Error in modeling_help_body: {e}")
        session.reply("I encountered an issue while preparing guidance. Could you try again?")

modeling_help_state.set_body(modeling_help_body)

def clarify_diagram_type_body(session: Session):
    """Ask user to clarify the diagram type when it cannot be determined.
    
    With unified JSON messages, this processes the request directly.
    """
    context = extract_modeling_context(session)
    if not context:
        session.reply("I need to know which diagram type you'd like. Please specify: Class, Object, StateMachine, or Agent diagram.")
        return

    actual_message = context['actual_message']

    # Process the clarification request directly (unified JSON contains all context)
    clarification_prompt = f"""I'd like to help you with: "{actual_message}"

However, I need to know which type of UML diagram you'd like to create:

📊 **ClassDiagram** - For classes, attributes, methods, and relationships
📦 **ObjectDiagram** - For object instances and their links
🔄 **StateMachineDiagram** - For states, transitions, and events
🤖 **AgentDiagram** - For agents, beliefs, goals, and messages

Please specify the diagram type, or rephrase your request with more context.

For example:
- "Create a User class" → ClassDiagram
- "Create a state machine for login" → StateMachineDiagram
- "Create an agent for customer service" → AgentDiagram"""

    try:
        session.reply(clarification_prompt)
    except Exception as e:
        logger.error(f"Error in clarify_diagram_type_body: {e}")
        session.reply("I need to know which diagram type you'd like. Please specify: Class, Object, StateMachine, or Agent diagram.")

clarify_diagram_type_state.set_body(clarify_diagram_type_body)


def add_unified_transitions(state, intents_map, fallback_state):
    """Add both text and JSON event transitions for a state.
    
    Args:
        state: The state to add transitions to
        intents_map: Dict mapping intent objects to destination states
        fallback_state: State to go to when no intent matches
    """
    # Text event transitions (backward compatibility)
    for intent, dest_state in intents_map.items():
        state.when_intent_matched(intent).go_to(dest_state)
    
    # JSON event transitions (unified messages)
    for intent, dest_state in intents_map.items():
        state.when_event(ReceiveJSONEvent())\
            .with_condition(json_intent_matches, {'intent_name': intent.name})\
            .go_to(dest_state)
    
    # Fallback transitions
    state.when_event(ReceiveJSONEvent())\
        .with_condition(json_no_intent_matched)\
        .go_to(fallback_state)
    state.when_no_intent_matched().go_to(fallback_state)


# Transitions from create_single_element state
add_unified_transitions(create_single_element_state, {
    create_single_element_intent: create_single_element_state,
    create_complete_system_intent: create_complete_system_state,
    modify_model_intent: modify_model_state,
    modeling_help_intent: modeling_help_state,
    hello_intent: greetings_state
}, create_single_element_state)

# Transitions from create_complete_system state
add_unified_transitions(create_complete_system_state, {
    create_single_element_intent: create_single_element_state,
    create_complete_system_intent: create_complete_system_state,
    modify_model_intent: modify_model_state,
    modeling_help_intent: modeling_help_state,
    hello_intent: greetings_state
}, create_complete_system_state)

# Transitions from modify state
add_unified_transitions(modify_model_state, {
    create_single_element_intent: create_single_element_state,
    create_complete_system_intent: create_complete_system_state,
    modify_model_intent: modify_model_state,
    modeling_help_intent: modeling_help_state,
    hello_intent: greetings_state
}, modify_model_state)

# Transitions from modeling help state
add_unified_transitions(modeling_help_state, {
    create_single_element_intent: create_single_element_state,
    create_complete_system_intent: create_complete_system_state,
    modify_model_intent: modify_model_state,
    modeling_help_intent: modeling_help_state,
    hello_intent: greetings_state
}, modeling_help_state)

# Transitions from clarify_diagram_type state
add_unified_transitions(clarify_diagram_type_state, {
    create_single_element_intent: create_single_element_state,
    create_complete_system_intent: create_complete_system_state,
    modify_model_intent: modify_model_state,
    modeling_help_intent: modeling_help_state,
    hello_intent: greetings_state
}, clarify_diagram_type_state)


# UML RAG STATE BODY

def uml_rag_body(session: Session):
    """Answer UML specification questions using RAG.
    
    With unified JSON messages, this extracts the message and processes it.
    """
    # Mark that we're handling a RAG question
    session.set('last_matched_intent', 'uml_spec_intent')
    
    # Get the user message (works for both text and JSON events)
    user_message = get_user_message(session)
    
    if not user_message:
        session.reply("Please ask a question about UML specifications.")
        return
    
    if uml_rag is None:
        # Fallback if RAG is not initialized
        fallback_response = gpt.predict(
            f"""You are a UML specification expert. Answer the following question about UML:

{user_message}

Provide accurate information based on UML 2.x specifications. Be precise and reference specific UML concepts when applicable."""
        )
        session.reply(fallback_response)
    else:
        try:
            rag_message: RAGMessage = session.run_rag(user_message)
            # Send only the answer text, not the full RAG JSON
            session.reply(rag_message.answer)
        except Exception as e:
            logger.error(f"Error in uml_rag_body: {e}")
            # Fallback to LLM if RAG fails
            fallback_response = gpt.predict(
                f"""You are a UML specification expert. Answer the following question about UML:

{user_message}

Provide accurate information based on UML 2.x specifications."""
            )
            session.reply(fallback_response)

uml_rag_state.set_body(uml_rag_body)

# Transitions from uml_rag_state (with unified JSON support)
add_unified_transitions(uml_rag_state, {
    create_single_element_intent: create_single_element_state,
    create_complete_system_intent: create_complete_system_state,
    modify_model_intent: modify_model_state,
    modeling_help_intent: modeling_help_state,
    uml_spec_intent: uml_rag_state,
    hello_intent: greetings_state
}, greetings_state)


# RUN APPLICATION
if __name__ == '__main__':
    agent.run()