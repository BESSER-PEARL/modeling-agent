"""
File Conversion Handler
Converts uploaded files into diagram specifications that can be injected
into the editor.  The handler automatically detects the best-fit diagram
type from the file content.

Supported file types:
  - PlantUML (.puml, .plantuml, .pu, .txt containing @startuml)
  - XMI / UML interchange (.xmi, .uml, .ecore — standard UML/EMF exchange formats)
  - Knowledge Graph (.ttl, .rdf, .owl, .json with KG structure)
  - Images (.png, .jpg, .jpeg, .gif, .webp — UML diagram photos/screenshots)
  - PDF (.pdf — text-based or scanned/diagram PDFs)
  - Generic text files — analyzed by the LLM to extract any diagram

Supported target diagram types:
  - ClassDiagram (classes, attributes, methods, relationships)
  - StateMachineDiagram (states, transitions)
  - ObjectDiagram (objects with attribute values, links)
  - AgentDiagram (intents, states, transitions for chatbot agents)
"""

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── File type detection ────────────────────────────────────────────────────────

PLANTUML_EXTENSIONS = {".puml", ".plantuml", ".pu"}
KG_EXTENSIONS = {".ttl", ".rdf", ".owl", ".n3", ".nt", ".nq", ".trig", ".jsonld"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
PDF_EXTENSIONS = {".pdf"}
XMI_EXTENSIONS = {".xmi", ".uml", ".ecore"}
IMAGE_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Diagram types the file converter can produce
CONVERTIBLE_DIAGRAM_TYPES = {
    "ClassDiagram",
    "StateMachineDiagram",
    "ObjectDiagram",
    "AgentDiagram",
}

# ── Per-diagram-type JSON schemas ──────────────────────────────────────────────

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
}"""

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

ALL_SCHEMAS = f"""
=== ClassDiagram (for entity/data models, schemas, class structures) ===
{CLASS_DIAGRAM_SCHEMA}

=== StateMachineDiagram (for workflows, processes, state transitions, protocols) ===
{STATE_MACHINE_SCHEMA}

=== ObjectDiagram (for specific instances/examples with concrete attribute values) ===
{OBJECT_DIAGRAM_SCHEMA}

=== AgentDiagram (for chatbot agents, intent/response flows, conversational AI) ===
{AGENT_DIAGRAM_SCHEMA}
"""


def detect_file_type(filename: str, content_text: Optional[str] = None) -> str:
    """Detect file type from extension and/or content.

    Returns one of: 'plantuml', 'knowledge_graph', 'image', 'unknown'.
    """
    ext = ""
    if filename:
        dot_pos = filename.rfind(".")
        if dot_pos >= 0:
            ext = filename[dot_pos:].lower()

    # Extension-based detection
    if ext in PLANTUML_EXTENSIONS:
        return "plantuml"
    if ext in KG_EXTENSIONS:
        return "knowledge_graph"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in XMI_EXTENSIONS:
        return "xmi"

    # Content-based detection for .txt or extensionless files
    if content_text:
        stripped = content_text.strip()
        if "@startuml" in stripped:
            return "plantuml"
        # Turtle/N3 detection
        if stripped.startswith("@prefix") or stripped.startswith("@base"):
            return "knowledge_graph"
        # XMI/UML XML detection
        if stripped.startswith("<?xml") and ("xmi:" in stripped or "uml:" in stripped or "XMI" in stripped):
            return "xmi"
        # RDF/XML detection
        if stripped.startswith("<?xml") and ("rdf:RDF" in stripped or "owl:" in stripped):
            return "knowledge_graph"
        # JSON-LD or Neo4j JSON detection
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    if "@context" in parsed or "@graph" in parsed:
                        return "knowledge_graph"
                    if "nodes" in parsed or "relationships" in parsed:
                        return "knowledge_graph"
                elif isinstance(parsed, list) and len(parsed) > 0:
                    first = parsed[0]
                    if isinstance(first, dict) and ("type" in first or "@type" in first):
                        return "knowledge_graph"
            except json.JSONDecodeError:
                pass

    # .json files that aren't KG could be many things
    if ext == ".json":
        return "knowledge_graph"

    return "unknown"


def detect_plantuml_diagram_type(content: str) -> str:
    """Detect the target diagram type from PlantUML syntax keywords.

    Inspects the PlantUML text for diagram-type-specific keywords and returns
    the best-fit diagram type string.
    """
    lower = content.lower()

    # State machine indicators (check before class — `state` keyword is specific)
    state_keywords = ["[*]", "state ", "-->", "<<fork>>", "<<join>>", "<<choice>>"]
    class_keywords = ["class ", "interface ", "abstract ", "enum ", "<|--", "*--", "o--"]
    object_keywords = ["object ", ":"]

    has_state = any(kw in lower for kw in state_keywords)
    has_class = any(kw in lower for kw in class_keywords)
    has_object = any(kw in lower for kw in object_keywords)

    # If it has class keywords, it's most likely a class diagram
    if has_class and not has_state:
        return "ClassDiagram"
    # If it has state keywords but no class keywords
    if has_state and not has_class:
        return "StateMachineDiagram"
    # If it has object keywords
    if has_object and not has_class and not has_state:
        return "ObjectDiagram"
    # Both class and state → let LLM decide, but default to class
    if has_class:
        return "ClassDiagram"
    if has_state:
        return "StateMachineDiagram"

    # No clear indicators — default to ClassDiagram
    return "ClassDiagram"


def _get_mime_type(filename: str) -> str:
    """Get MIME type for an image file."""
    ext = ""
    if filename:
        dot_pos = filename.rfind(".")
        if dot_pos >= 0:
            ext = filename[dot_pos:].lower()
    return IMAGE_MIME_MAP.get(ext, "image/png")


# ── LLM conversion prompts ────────────────────────────────────────────────────

def _build_conversion_prompt(source_label: str, target_diagram_type: str) -> str:
    """Build a diagram-type-specific conversion prompt."""
    schema = _get_schema_for_type(target_diagram_type)
    type_label = _get_type_label(target_diagram_type)

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


def _build_auto_detect_prompt(source_label: str) -> str:
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
        "When in doubt, prefer ClassDiagram — it is the most common and versatile.\n"
        "- StateMachineDiagram: ONLY when the content explicitly defines named states and "
        "transitions between them (e.g., 'Idle -> Running -> Stopped'). "
        "Do NOT choose this just because a document mentions a process or workflow.\n"
        "- ObjectDiagram: specific instances with concrete values (e.g., 'user1 : User, name=\"Alice\"')\n"
        "- AgentDiagram: chatbot flows, intent-response patterns, conversational AI\n\n"
        "IMPORTANT RULES:\n"
        "1. You MUST include the \"diagramType\" field in your response\n"
        "2. Extract ALL relevant elements from the source content\n"
        "3. Use proper type names: String, int, boolean, double, Date, float, long\n"
        "4. Return ONLY valid JSON, no explanations or markdown\n"
    )


def _build_image_prompt(target_diagram_type: Optional[str] = None) -> str:
    """Build Vision API prompt for image-based conversion."""
    if target_diagram_type and target_diagram_type in CONVERTIBLE_DIAGRAM_TYPES:
        schema = _get_schema_for_type(target_diagram_type)
        type_label = _get_type_label(target_diagram_type)
        return (
            f"You are a UML/modeling expert. Analyze this image of a {type_label} "
            f"and convert it into a structured JSON specification.\n\n"
            f"Return ONLY a JSON object with this exact structure:\n{schema}\n\n"
            "IMPORTANT: Extract EVERY element visible in the diagram. "
            "Keep names exactly as shown. Include the \"diagramType\" field. "
            "Return ONLY valid JSON."
        )

    # Auto-detect from image
    return (
        "You are a UML/modeling expert. Analyze this image of a UML diagram.\n\n"
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
    }.get(diagram_type, CLASS_DIAGRAM_SCHEMA)


def _get_type_label(diagram_type: str) -> str:
    """Return a human-readable label for a diagram type."""
    return {
        "ClassDiagram": "UML Class Diagram",
        "StateMachineDiagram": "UML State Machine Diagram",
        "ObjectDiagram": "UML Object Diagram",
        "AgentDiagram": "Agent / Chatbot Diagram",
    }.get(diagram_type, "UML Diagram")


# ── Conversion logic ──────────────────────────────────────────────────────────

def convert_file_to_diagram_spec(
    file_content_b64: str,
    filename: str,
    llm_predict: callable,
    llm_vision_predict: Optional[callable] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert an uploaded file to the best-fit diagram specification.

    Automatically detects the file type AND the target diagram type from
    the content, then produces the correctly-shaped system spec.

    Args:
        file_content_b64: Base64-encoded file content.
        filename: Original filename (used for type detection).
        llm_predict: LLM predict function for text-based conversions.
        llm_vision_predict: Optional vision-capable LLM predict for images.
        openai_api_key: OpenAI API key for direct vision API calls (fallback).

    Returns:
        A dict with action='inject_complete_system' and the system spec,
        or action='agent_error' on failure.
    """
    try:
        raw_bytes = base64.b64decode(file_content_b64)
    except Exception as e:
        logger.error(f"[FileConversion] Failed to decode base64 content: {e}")
        return _error_response("Could not decode the uploaded file. Please try again.")

    # Attempt to decode as text for non-image files
    content_text = None
    try:
        content_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    file_type = detect_file_type(filename, content_text)
    logger.info(f"[FileConversion] Detected file type: {file_type} for file: {filename}")

    if file_type == "plantuml":
        return _convert_plantuml(content_text, filename, llm_predict)
    elif file_type == "knowledge_graph":
        return _convert_knowledge_graph(content_text, filename, llm_predict)
    elif file_type == "image":
        return _convert_image(file_content_b64, filename, llm_predict, openai_api_key)
    elif file_type == "pdf":
        return _convert_pdf(raw_bytes, filename, llm_predict, openai_api_key)
    elif file_type == "xmi":
        return _convert_xmi(content_text, filename, llm_predict)
    else:
        # Try to convert as text if we have text content
        if content_text:
            return _convert_generic_text(content_text, filename, llm_predict)
        return _error_response(
            f"Could not determine the file type for '{filename}'. "
            "Supported formats: PlantUML (.puml), XMI/UML (.xmi, .uml, .ecore), "
            "Knowledge Graphs (.ttl, .rdf, .owl, .json), "
            "PDF documents (.pdf), or images of UML diagrams (.png, .jpg)."
        )


# Keep the old name as an alias for backward compatibility
convert_file_to_class_spec = convert_file_to_diagram_spec


def _convert_plantuml(
    content: str, filename: str, llm_predict: callable,
) -> Dict[str, Any]:
    """Convert PlantUML text to the best-fit diagram spec via LLM."""
    diagram_type = detect_plantuml_diagram_type(content)
    prompt = _build_conversion_prompt("PlantUML diagram", diagram_type)
    prompt += f"\n\nPlantUML Code:\n```\n{content}\n```"
    return _run_llm_conversion(prompt, filename, "PlantUML", diagram_type, llm_predict)


def _convert_knowledge_graph(
    content: str, filename: str, llm_predict: callable,
) -> Dict[str, Any]:
    """Convert a Knowledge Graph file to the best-fit diagram spec via LLM.

    Knowledge graphs typically map to class diagrams, but the LLM may detect
    state/process patterns and choose a different type.
    """
    if len(content) > 30_000:
        content = content[:30_000] + "\n\n... [truncated — file too large, showing first 30KB]"
    prompt = _build_auto_detect_prompt("Knowledge Graph data")
    prompt += f"\n\nKnowledge Graph Data:\n```\n{content}\n```"
    return _run_llm_conversion(prompt, filename, "Knowledge Graph", None, llm_predict)


def _convert_generic_text(
    content: str, filename: str, llm_predict: callable,
) -> Dict[str, Any]:
    """Attempt to convert any text file to the best-fit diagram via LLM."""
    if len(content) > 30_000:
        content = content[:30_000] + "\n\n... [truncated]"
    prompt = _build_auto_detect_prompt(f"text file ('{filename}')")
    prompt += f"\n\nFile Content:\n```\n{content}\n```"
    return _run_llm_conversion(prompt, filename, "text file", None, llm_predict)


def _convert_xmi(
    content: str, filename: str, llm_predict: callable,
) -> Dict[str, Any]:
    """Convert an XMI/UML interchange file to the best-fit diagram spec via LLM."""
    if len(content) > 30_000:
        content = content[:30_000] + "\n\n... [truncated — file too large, showing first 30KB]"
    prompt = _build_auto_detect_prompt("XMI/UML interchange file")
    prompt += (
        "\n\nIMPORTANT: This is an XMI (XML Metadata Interchange) file — a standard "
        "format for exchanging UML models. Parse the XML elements and attributes "
        "to extract classes, properties, associations, state machines, etc.\n\n"
        f"XMI Content:\n```xml\n{content}\n```"
    )
    return _run_llm_conversion(prompt, filename, "XMI", None, llm_predict)


def _convert_pdf(
    pdf_bytes: bytes,
    filename: str,
    llm_predict: callable,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a PDF document to the best-fit diagram spec.

    Strategy:
      1. Extract text from all pages using PyMuPDF.
      2. If enough text is found (>50 chars), treat it as a text-based PDF
         and run LLM auto-detect conversion on the extracted text.
      3. If the PDF is mostly images/diagrams (little text), render the first
         pages to images and run vision-based conversion.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _error_response(
            "PDF processing is not available. The PyMuPDF library is missing. "
            "Please install it with: pip install PyMuPDF"
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"[FileConversion] Failed to open PDF '{filename}': {e}")
        return _error_response(
            f"Could not open the PDF file '{filename}'. It may be corrupted or password-protected."
        )

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        return _error_response(f"The PDF file '{filename}' has no pages.")

    # Extract all content from the PDF upfront, then close the doc.
    # Using try/finally ensures the doc is always closed even on unexpected errors.
    full_text = ""
    images_b64 = []
    try:
        # Extract text from all pages (cap at 20 pages to stay within LLM limits)
        text_parts = []
        max_pages = min(page_count, 20)
        for i in range(max_pages):
            page_text = doc[i].get_text("text").strip()
            if page_text:
                text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
        full_text = "\n\n".join(text_parts)

        # If low text, render pages to images for vision analysis
        if len(full_text) <= 50 and openai_api_key:
            render_pages = min(page_count, 3)
            for i in range(render_pages):
                try:
                    pix = doc[i].get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    images_b64.append(base64.b64encode(img_bytes).decode("ascii"))
                except Exception as e:
                    logger.warning(f"[FileConversion] Failed to render PDF page {i + 1}: {e}")
    finally:
        doc.close()

    # If we got meaningful text, use text-based conversion
    if len(full_text) > 50:
        if len(full_text) > 30_000:
            full_text = full_text[:30_000] + "\n\n... [truncated — showing first 30KB]"
        prompt = _build_auto_detect_prompt(f"PDF document ('{filename}', {page_count} page(s))")
        prompt += f"\n\nExtracted PDF Content:\n```\n{full_text}\n```"
        return _run_llm_conversion(prompt, filename, "PDF", None, llm_predict)

    # Low text content — PDF likely contains diagrams/images.
    if not openai_api_key:
        return _error_response(
            f"The PDF '{filename}' appears to contain diagrams/images rather than text. "
            "Image-based PDF conversion requires an OpenAI API key with vision capabilities."
        )

    if not images_b64:
        return _error_response(
            f"Could not extract content from the PDF '{filename}'. "
            "The file may be empty or in an unsupported format."
        )

    # Send page images to the vision API
    vision_prompt = _build_image_prompt()  # auto-detect from image
    content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": vision_prompt}]
    for img_b64 in images_b64:
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })

    try:
        import requests as http_requests

        response = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}",
            },
            json={
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": content_blocks}],
                "max_tokens": 8192,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[FileConversion] Vision API call failed for PDF: {e}")
        return _error_response(
            f"Failed to process the diagram images from PDF '{filename}'. "
            "Please make sure the PDF contains clear UML diagrams."
        )

    return _parse_llm_response(raw_text, filename, "PDF (image)", expected_type=None)


def _convert_image(
    image_b64: str,
    filename: str,
    llm_predict: callable,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert an image of a UML diagram to the best-fit diagram spec."""
    if not openai_api_key:
        return _error_response(
            "Image conversion requires an OpenAI API key with vision capabilities. "
            "Please configure the API key in the agent settings."
        )

    mime_type = _get_mime_type(filename)
    vision_prompt = _build_image_prompt()  # auto-detect from image

    try:
        import requests as http_requests

        response = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}",
            },
            json={
                "model": "gpt-4.1",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 8192,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[FileConversion] Vision API call failed: {e}")
        return _error_response(
            "Failed to process the image. Please make sure the image contains a clear UML diagram."
        )

    return _parse_llm_response(raw_text, filename, "image", expected_type=None)


def _run_llm_conversion(
    prompt: str,
    filename: str,
    source_label: str,
    expected_type: Optional[str],
    llm_predict: callable,
) -> Dict[str, Any]:
    """Run LLM conversion for text-based files."""
    try:
        raw_response = llm_predict(prompt)
    except Exception as e:
        logger.error(f"[FileConversion] LLM prediction failed for {source_label}: {e}")
        return _error_response(
            f"Failed to process the {source_label} file. The AI model encountered an error."
        )
    return _parse_llm_response(raw_response, filename, source_label, expected_type)


def _parse_llm_response(
    raw_response: str,
    filename: str,
    source_label: str,
    expected_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse the LLM response, detect diagram type, and validate the spec."""
    # Clean markdown code fences if present
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        spec = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"[FileConversion] Failed to parse LLM JSON for {source_label}: {e}")
        logger.debug(f"[FileConversion] Raw response: {raw_response[:500]!r}")
        return _error_response(
            f"The AI couldn't produce a valid specification from the {source_label} file. "
            "Please try again or simplify the input."
        )

    if not isinstance(spec, dict):
        return _error_response(f"Unexpected response format from {source_label} conversion.")

    # Determine the diagram type from the spec or the expected type
    diagram_type = _resolve_diagram_type(spec, expected_type)
    logger.info(f"[FileConversion] Resolved diagram type: {diagram_type} for {filename}")

    # Validate and build the response based on diagram type
    if diagram_type == "StateMachineDiagram":
        return _validate_state_machine_spec(spec, filename, source_label)
    elif diagram_type == "ObjectDiagram":
        return _validate_object_diagram_spec(spec, filename, source_label)
    elif diagram_type == "AgentDiagram":
        return _validate_agent_diagram_spec(spec, filename, source_label)
    else:
        return _validate_class_diagram_spec(spec, filename, source_label)


def _resolve_diagram_type(spec: Dict[str, Any], expected_type: Optional[str]) -> str:
    """Determine the diagram type from the LLM response and/or expected type."""
    # If the LLM included a diagramType field, trust it
    llm_type = spec.get("diagramType", "")
    if isinstance(llm_type, str) and llm_type in CONVERTIBLE_DIAGRAM_TYPES:
        return llm_type

    # If we had an expected type from pre-detection (e.g., PlantUML keywords)
    if expected_type and expected_type in CONVERTIBLE_DIAGRAM_TYPES:
        return expected_type

    # Infer from the shape of the spec
    if "states" in spec and isinstance(spec["states"], list) and len(spec.get("states", [])) > 0:
        first = spec["states"][0] if spec["states"] else {}
        if isinstance(first, dict):
            # Agent diagrams have states with "replies" or "type"
            if "replies" in first or first.get("type") == "state":
                if "intents" in spec:
                    return "AgentDiagram"
            # State machine states have "stateName" and "stateType"
            if "stateName" in first or "stateType" in first:
                return "StateMachineDiagram"

    if "objects" in spec and isinstance(spec["objects"], list) and len(spec["objects"]) > 0:
        return "ObjectDiagram"

    if "intents" in spec and isinstance(spec["intents"], list) and len(spec["intents"]) > 0:
        return "AgentDiagram"

    # Default to ClassDiagram
    return "ClassDiagram"


# ── Per-type validators ───────────────────────────────────────────────────────

def _validate_class_diagram_spec(
    spec: Dict[str, Any], filename: str, source_label: str,
) -> Dict[str, Any]:
    """Validate and build a ClassDiagram inject_complete_system response."""
    classes = spec.get("classes", [])
    if not isinstance(classes, list) or len(classes) == 0:
        return _error_response(
            f"No classes could be extracted from the {source_label} file. "
            "Please check that the file contains valid class/entity definitions."
        )

    relationships = spec.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    validated_classes = []
    for cls in classes:
        if not isinstance(cls, dict):
            continue
        class_name = cls.get("className")
        if not isinstance(class_name, str) or not class_name.strip():
            continue
        validated_classes.append({
            "className": class_name.strip(),
            "attributes": cls.get("attributes", []) if isinstance(cls.get("attributes"), list) else [],
            "methods": cls.get("methods", []) if isinstance(cls.get("methods"), list) else [],
        })

    if not validated_classes:
        return _error_response(f"No valid classes found in the {source_label} file.")

    class_names = {c["className"] for c in validated_classes}
    validated_rels = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        source = rel.get("source", "")
        target = rel.get("target", "")
        if source in class_names and target in class_names:
            validated_rels.append(rel)
        else:
            logger.warning(
                f"[FileConversion] Dropping relationship {source}->{target}: "
                f"class not found in extracted classes"
            )

    system_spec = {"classes": validated_classes, "relationships": validated_rels}
    attr_count = sum(len(c.get("attributes", [])) for c in validated_classes)

    return _success_response(
        diagram_type="ClassDiagram",
        system_spec=system_spec,
        filename=filename,
        source_label=source_label,
        summary=f"{len(validated_classes)} class(es), {attr_count} attribute(s), "
                f"and {len(validated_rels)} relationship(s)",
    )


def _validate_state_machine_spec(
    spec: Dict[str, Any], filename: str, source_label: str,
) -> Dict[str, Any]:
    """Validate and build a StateMachineDiagram inject_complete_system response."""
    states = spec.get("states", [])
    if not isinstance(states, list) or len(states) == 0:
        return _error_response(
            f"No states could be extracted from the {source_label} file. "
            "Please check that the file contains state/workflow definitions."
        )

    transitions = spec.get("transitions", [])
    if not isinstance(transitions, list):
        transitions = []

    validated_states = []
    for state in states:
        if not isinstance(state, dict):
            continue
        state_name = state.get("stateName")
        if not isinstance(state_name, str) or not state_name.strip():
            continue
        validated_states.append({
            "stateName": state_name.strip(),
            "stateType": state.get("stateType", "regular") if isinstance(state.get("stateType"), str) else "regular",
            "entryAction": state.get("entryAction", "") if isinstance(state.get("entryAction"), str) else "",
            "exitAction": state.get("exitAction", "") if isinstance(state.get("exitAction"), str) else "",
            "doActivity": state.get("doActivity", "") if isinstance(state.get("doActivity"), str) else "",
        })

    if not validated_states:
        return _error_response(f"No valid states found in the {source_label} file.")

    state_names = {s["stateName"] for s in validated_states}
    validated_trans = []
    for trans in transitions:
        if not isinstance(trans, dict):
            continue
        source = trans.get("source", "")
        target = trans.get("target", "")
        # Allow [*] and special initial/final references
        source_ok = source in state_names or source in ("[*]", "initial", "Initial")
        target_ok = target in state_names or target in ("[*]", "final", "Final")
        if source_ok and target_ok:
            validated_trans.append(trans)
        else:
            logger.warning(
                f"[FileConversion] Dropping transition {source}->{target}: "
                f"state not found in extracted states"
            )

    system_spec = {"states": validated_states, "transitions": validated_trans}

    return _success_response(
        diagram_type="StateMachineDiagram",
        system_spec=system_spec,
        filename=filename,
        source_label=source_label,
        summary=f"{len(validated_states)} state(s) and {len(validated_trans)} transition(s)",
    )


def _validate_object_diagram_spec(
    spec: Dict[str, Any], filename: str, source_label: str,
) -> Dict[str, Any]:
    """Validate and build an ObjectDiagram inject_complete_system response."""
    objects = spec.get("objects", [])
    if not isinstance(objects, list) or len(objects) == 0:
        return _error_response(
            f"No objects could be extracted from the {source_label} file. "
            "Please check that the file contains object/instance definitions."
        )

    links = spec.get("links", [])
    if not isinstance(links, list):
        links = []

    validated_objects = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_name = obj.get("objectName")
        if not isinstance(obj_name, str) or not obj_name.strip():
            continue
        validated_objects.append({
            "objectName": obj_name.strip(),
            "className": obj.get("className", "") if isinstance(obj.get("className"), str) else "",
            "attributes": obj.get("attributes", []) if isinstance(obj.get("attributes"), list) else [],
        })

    if not validated_objects:
        return _error_response(f"No valid objects found in the {source_label} file.")

    obj_names = {o["objectName"] for o in validated_objects}
    validated_links = []
    for link in links:
        if not isinstance(link, dict):
            continue
        source = link.get("source", "")
        target = link.get("target", "")
        if source in obj_names and target in obj_names:
            validated_links.append(link)

    system_spec = {"objects": validated_objects, "links": validated_links}

    return _success_response(
        diagram_type="ObjectDiagram",
        system_spec=system_spec,
        filename=filename,
        source_label=source_label,
        summary=f"{len(validated_objects)} object(s) and {len(validated_links)} link(s)",
    )


def _validate_agent_diagram_spec(
    spec: Dict[str, Any], filename: str, source_label: str,
) -> Dict[str, Any]:
    """Validate and build an AgentDiagram inject_complete_system response."""
    intents = spec.get("intents", [])
    states = spec.get("states", [])
    transitions = spec.get("transitions", [])

    if not isinstance(intents, list):
        intents = []
    if not isinstance(states, list):
        states = []
    if not isinstance(transitions, list):
        transitions = []

    if len(intents) == 0 and len(states) == 0:
        return _error_response(
            f"No intents or states could be extracted from the {source_label} file. "
            "Please check that the file describes a chatbot or agent flow."
        )

    validated_intents = []
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        name = intent.get("intentName")
        if not isinstance(name, str) or not name.strip():
            continue
        validated_intents.append({
            "intentName": name.strip(),
            "trainingPhrases": intent.get("trainingPhrases", [])
                if isinstance(intent.get("trainingPhrases"), list) else [],
        })

    validated_states = []
    for state in states:
        if not isinstance(state, dict):
            continue
        name = state.get("stateName")
        if not isinstance(name, str) or not name.strip():
            continue
        validated_states.append({
            "stateName": name.strip(),
            "type": state.get("type", "state") if isinstance(state.get("type"), str) else "state",
            "replies": state.get("replies", []) if isinstance(state.get("replies"), list) else [],
        })

    system_spec = {
        "intents": validated_intents,
        "states": validated_states,
        "transitions": transitions,  # Pass through — handler will validate
    }

    parts = []
    if validated_intents:
        parts.append(f"{len(validated_intents)} intent(s)")
    if validated_states:
        parts.append(f"{len(validated_states)} state(s)")
    if transitions:
        parts.append(f"{len(transitions)} transition(s)")

    return _success_response(
        diagram_type="AgentDiagram",
        system_spec=system_spec,
        filename=filename,
        source_label=source_label,
        summary=", ".join(parts) if parts else "agent specification",
    )


# ── Response helpers ──────────────────────────────────────────────────────────

def _success_response(
    diagram_type: str,
    system_spec: Dict[str, Any],
    filename: str,
    source_label: str,
    summary: str,
) -> Dict[str, Any]:
    """Build a successful inject_complete_system response."""
    type_label = _get_type_label(diagram_type)
    message = f"Converted {source_label} file '{filename}' into a {type_label} with {summary}."
    logger.info(f"[FileConversion] {message}")

    return {
        "action": "inject_complete_system",
        "systemSpec": system_spec,
        "diagramType": diagram_type,
        "message": message,
    }


def _error_response(message: str) -> Dict[str, Any]:
    """Build a standardized error response."""
    return {
        "action": "agent_error",
        "code": "file_conversion_error",
        "message": message,
        "retryable": True,
    }
