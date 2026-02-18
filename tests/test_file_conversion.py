"""Tests for the file conversion handler."""

import base64
import json
import pytest

from handlers.file_conversion_handler import (
    convert_file_to_class_spec,
    convert_file_to_diagram_spec,
    detect_file_type,
    detect_plantuml_diagram_type,
    _resolve_diagram_type,
    CONVERTIBLE_DIAGRAM_TYPES,
)


# ── File type detection ──────────────────────────────────────────────────────


class TestDetectFileType:
    """Tests for detect_file_type()."""

    def test_plantuml_by_extension_puml(self):
        assert detect_file_type("diagram.puml") == "plantuml"

    def test_plantuml_by_extension_plantuml(self):
        assert detect_file_type("model.plantuml") == "plantuml"

    def test_plantuml_by_extension_pu(self):
        assert detect_file_type("test.pu") == "plantuml"

    def test_plantuml_by_content(self):
        content = "@startuml\nclass User {\n  +name: String\n}\n@enduml"
        assert detect_file_type("model.txt", content) == "plantuml"

    def test_kg_turtle_extension(self):
        assert detect_file_type("ontology.ttl") == "knowledge_graph"

    def test_kg_rdf_extension(self):
        assert detect_file_type("data.rdf") == "knowledge_graph"

    def test_kg_owl_extension(self):
        assert detect_file_type("ontology.owl") == "knowledge_graph"

    def test_kg_jsonld_extension(self):
        assert detect_file_type("data.jsonld") == "knowledge_graph"

    def test_kg_turtle_by_content(self):
        content = '@prefix ex: <http://example.org/> .\nex:Person a rdfs:Class .'
        assert detect_file_type("data.txt", content) == "knowledge_graph"

    def test_kg_rdf_xml_by_content(self):
        content = '<?xml version="1.0"?>\n<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        assert detect_file_type("data.txt", content) == "knowledge_graph"

    def test_kg_jsonld_by_content(self):
        content = '{"@context": "http://schema.org/", "@type": "Person", "name": "John"}'
        assert detect_file_type("data.txt", content) == "knowledge_graph"

    def test_kg_neo4j_json_by_content(self):
        content = '{"nodes": [{"id": 1}], "relationships": [{"type": "KNOWS"}]}'
        assert detect_file_type("export.txt", content) == "knowledge_graph"

    def test_json_defaults_to_kg(self):
        assert detect_file_type("data.json") == "knowledge_graph"

    def test_image_png(self):
        assert detect_file_type("diagram.png") == "image"

    def test_image_jpg(self):
        assert detect_file_type("photo.jpg") == "image"

    def test_image_jpeg(self):
        assert detect_file_type("photo.jpeg") == "image"

    def test_image_webp(self):
        assert detect_file_type("screenshot.webp") == "image"

    def test_unknown_extension(self):
        assert detect_file_type("data.xyz") == "unknown"

    def test_no_extension_no_content(self):
        assert detect_file_type("README") == "unknown"

    def test_empty_filename(self):
        assert detect_file_type("") == "unknown"

    def test_generic_text_fallback(self):
        assert detect_file_type("notes.txt", "just some random text") == "unknown"


# ── File conversion ──────────────────────────────────────────────────────────


def _make_b64(text: str) -> str:
    """Encode text as base64."""
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def _mock_llm_predict(prompt: str) -> str:
    """Mock LLM that returns a valid system spec JSON."""
    return json.dumps({
        "classes": [
            {
                "className": "User",
                "attributes": [
                    {"name": "id", "type": "int", "visibility": "private"},
                    {"name": "username", "type": "String", "visibility": "public"},
                    {"name": "email", "type": "String", "visibility": "public"},
                ],
                "methods": [],
            },
            {
                "className": "Order",
                "attributes": [
                    {"name": "orderId", "type": "int", "visibility": "private"},
                    {"name": "total", "type": "double", "visibility": "public"},
                ],
                "methods": [
                    {
                        "name": "calculateTotal",
                        "returnType": "double",
                        "visibility": "public",
                        "parameters": [],
                    }
                ],
            },
        ],
        "relationships": [
            {
                "type": "association",
                "source": "User",
                "target": "Order",
                "name": "places",
                "sourceMultiplicity": "1",
                "targetMultiplicity": "*",
            }
        ],
    })


def _mock_llm_predict_bad_json(prompt: str) -> str:
    """Mock LLM that returns invalid JSON."""
    return "This is not JSON at all!"


def _mock_llm_predict_empty_classes(prompt: str) -> str:
    """Mock LLM that returns a spec with no classes."""
    return json.dumps({"classes": [], "relationships": []})


def _mock_llm_predict_raises(prompt: str) -> str:
    """Mock LLM that raises an exception."""
    raise RuntimeError("API rate limit exceeded")


class TestConvertFilePlantuml:
    """Test PlantUML file conversion."""

    def test_plantuml_successful_conversion(self):
        puml = "@startuml\nclass User {\n  +name: String\n}\n@enduml"
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(puml),
            filename="model.puml",
            llm_predict=_mock_llm_predict,
        )
        assert result["action"] == "inject_complete_system"
        assert result["diagramType"] == "ClassDiagram"
        assert len(result["systemSpec"]["classes"]) == 2
        assert len(result["systemSpec"]["relationships"]) == 1
        assert "2 class" in result["message"]
        assert "model.puml" in result["message"]

    def test_plantuml_txt_extension_detected_by_content(self):
        puml = "@startuml\nclass Foo {}\n@enduml"
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(puml),
            filename="model.txt",
            llm_predict=_mock_llm_predict,
        )
        assert result["action"] == "inject_complete_system"


class TestConvertFileKnowledgeGraph:
    """Test Knowledge Graph file conversion."""

    def test_turtle_file_conversion(self):
        ttl = "@prefix ex: <http://example.org/> .\nex:Person a rdfs:Class ."
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(ttl),
            filename="ontology.ttl",
            llm_predict=_mock_llm_predict,
        )
        assert result["action"] == "inject_complete_system"
        assert result["diagramType"] == "ClassDiagram"

    def test_rdf_file_conversion(self):
        rdf = '<?xml version="1.0"?>\n<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"></rdf:RDF>'
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(rdf),
            filename="data.rdf",
            llm_predict=_mock_llm_predict,
        )
        assert result["action"] == "inject_complete_system"


class TestConvertFileImage:
    """Test image file conversion."""

    def test_image_without_api_key_returns_error(self):
        # A small 1x1 PNG
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        result = convert_file_to_class_spec(
            file_content_b64=png_b64,
            filename="diagram.png",
            llm_predict=_mock_llm_predict,
            openai_api_key=None,
        )
        assert result["action"] == "agent_error"
        assert "vision" in result["message"].lower() or "api key" in result["message"].lower()


class TestConvertFileEdgeCases:
    """Test error handling and edge cases."""

    def test_invalid_base64_returns_error(self):
        result = convert_file_to_class_spec(
            file_content_b64="not-valid-base64!!!",
            filename="model.puml",
            llm_predict=_mock_llm_predict,
        )
        assert result["action"] == "agent_error"

    def test_bad_json_from_llm_returns_error(self):
        puml = "@startuml\nclass User {}\n@enduml"
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(puml),
            filename="model.puml",
            llm_predict=_mock_llm_predict_bad_json,
        )
        assert result["action"] == "agent_error"

    def test_empty_classes_returns_error(self):
        puml = "@startuml\n@enduml"
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(puml),
            filename="model.puml",
            llm_predict=_mock_llm_predict_empty_classes,
        )
        assert result["action"] == "agent_error"

    def test_llm_exception_returns_error(self):
        puml = "@startuml\nclass User {}\n@enduml"
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(puml),
            filename="model.puml",
            llm_predict=_mock_llm_predict_raises,
        )
        assert result["action"] == "agent_error"

    def test_unknown_file_type_with_text_attempts_generic(self):
        content = "Entity User has name, email. Entity Order has total."
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(content),
            filename="schema.txt",
            llm_predict=_mock_llm_predict,
        )
        # Should attempt generic text conversion and succeed with mock
        assert result["action"] == "inject_complete_system"

    def test_unknown_file_type_binary_returns_error(self):
        # Binary content that can't be decoded as UTF-8
        binary_b64 = base64.b64encode(bytes(range(128, 256))).decode("utf-8")
        result = convert_file_to_class_spec(
            file_content_b64=binary_b64,
            filename="data.bin",
            llm_predict=_mock_llm_predict,
        )
        assert result["action"] == "agent_error"

    def test_relationship_with_missing_class_is_dropped(self):
        """Relationships referencing non-existent classes should be dropped."""
        def mock_predict_bad_rel(prompt: str) -> str:
            return json.dumps({
                "classes": [
                    {"className": "User", "attributes": [], "methods": []},
                ],
                "relationships": [
                    {
                        "type": "association",
                        "source": "User",
                        "target": "NonExistentClass",
                        "name": "broken",
                    }
                ],
            })

        result = convert_file_to_class_spec(
            file_content_b64=_make_b64("@startuml\nclass User {}\n@enduml"),
            filename="model.puml",
            llm_predict=mock_predict_bad_rel,
        )
        assert result["action"] == "inject_complete_system"
        assert len(result["systemSpec"]["relationships"]) == 0

    def test_message_includes_counts(self):
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64("@startuml\nclass Test {}\n@enduml"),
            filename="test.puml",
            llm_predict=_mock_llm_predict,
        )
        assert "2 class" in result["message"]
        assert "1 relationship" in result["message"]
        assert "5 attribute" in result["message"]


# ── Protocol integration (FileAttachment parsing) ────────────────────────────


class TestFileAttachmentParsing:
    """Test that the protocol adapter correctly parses file attachments."""

    def test_parse_v2_payload_with_attachments(self):
        from protocol.adapters import parse_v2_payload

        raw = {
            "action": "user_message",
            "protocolVersion": "2.0",
            "message": "Convert this file",
            "context": {"activeDiagramType": "ClassDiagram"},
            "attachments": [
                {
                    "filename": "model.puml",
                    "content": _make_b64("@startuml\nclass A {}\n@enduml"),
                    "mimeType": "text/plain",
                }
            ],
        }
        request = parse_v2_payload(raw)
        assert request.has_attachments
        assert len(request.attachments) == 1
        assert request.attachments[0].filename == "model.puml"
        assert request.attachments[0].mime_type == "text/plain"
        assert len(request.attachments[0].content_b64) > 0

    def test_parse_v2_payload_without_attachments(self):
        from protocol.adapters import parse_v2_payload

        raw = {
            "action": "user_message",
            "protocolVersion": "2.0",
            "message": "Create a User class",
            "context": {"activeDiagramType": "ClassDiagram"},
        }
        request = parse_v2_payload(raw)
        assert not request.has_attachments
        assert len(request.attachments) == 0

    def test_parse_v2_payload_with_invalid_attachments(self):
        from protocol.adapters import parse_v2_payload

        raw = {
            "action": "user_message",
            "protocolVersion": "2.0",
            "message": "Hello",
            "context": {},
            "attachments": [
                {"filename": "no_content.txt"},  # Missing content field
                "not_a_dict",  # Invalid type
                {"content": _make_b64("valid"), "filename": "ok.txt"},  # Valid
            ],
        }
        request = parse_v2_payload(raw)
        assert len(request.attachments) == 1
        assert request.attachments[0].filename == "ok.txt"

    def test_parse_v2_payload_attachments_not_a_list(self):
        from protocol.adapters import parse_v2_payload

        raw = {
            "action": "user_message",
            "protocolVersion": "2.0",
            "message": "Hello",
            "context": {},
            "attachments": "not_a_list",
        }
        request = parse_v2_payload(raw)
        assert not request.has_attachments


# ── PlantUML diagram type detection ──────────────────────────────────────────


class TestDetectPlantUMLDiagramType:
    """Tests for detect_plantuml_diagram_type()."""

    def test_class_diagram_keywords(self):
        puml = "@startuml\nclass User {\n  +name: String\n}\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "ClassDiagram"

    def test_state_machine_keywords(self):
        puml = "@startuml\n[*] --> Idle\nIdle --> Active : start\nActive --> [*]\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "StateMachineDiagram"

    def test_state_keyword(self):
        puml = "@startuml\nstate Idle\nstate Active\nIdle --> Active\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "StateMachineDiagram"

    def test_object_diagram_keywords(self):
        puml = "@startuml\nobject user1\nobject order1\nuser1 : name = Alice\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "ObjectDiagram"

    def test_inheritance_is_class(self):
        puml = "@startuml\nAnimal <|-- Dog\nAnimal <|-- Cat\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "ClassDiagram"

    def test_composition_is_class(self):
        puml = "@startuml\nCar *-- Engine\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "ClassDiagram"

    def test_no_keywords_defaults_to_class(self):
        puml = "@startuml\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "ClassDiagram"

    def test_mixed_class_and_state_prefers_class(self):
        puml = "@startuml\nclass Foo {}\n[*] --> Foo\n@enduml"
        assert detect_plantuml_diagram_type(puml) == "ClassDiagram"


# ── Diagram type resolution ─────────────────────────────────────────────────


class TestResolveDiagramType:
    """Tests for _resolve_diagram_type()."""

    def test_explicit_diagram_type_field(self):
        spec = {"diagramType": "StateMachineDiagram", "states": []}
        assert _resolve_diagram_type(spec, None) == "StateMachineDiagram"

    def test_expected_type_fallback(self):
        spec = {"classes": []}
        assert _resolve_diagram_type(spec, "ObjectDiagram") == "ObjectDiagram"

    def test_infer_state_machine_from_shape(self):
        spec = {
            "states": [{"stateName": "Idle", "stateType": "initial"}],
            "transitions": [],
        }
        assert _resolve_diagram_type(spec, None) == "StateMachineDiagram"

    def test_infer_object_diagram_from_shape(self):
        spec = {
            "objects": [{"objectName": "user1", "className": "User"}],
            "links": [],
        }
        assert _resolve_diagram_type(spec, None) == "ObjectDiagram"

    def test_infer_agent_diagram_from_intents(self):
        spec = {
            "intents": [{"intentName": "greet"}],
            "states": [{"stateName": "welcome", "type": "state", "replies": []}],
        }
        assert _resolve_diagram_type(spec, None) == "AgentDiagram"

    def test_default_to_class_diagram(self):
        spec = {"something": "else"}
        assert _resolve_diagram_type(spec, None) == "ClassDiagram"

    def test_explicit_type_overrides_shape(self):
        spec = {
            "diagramType": "ObjectDiagram",
            "states": [{"stateName": "A", "stateType": "initial"}],
        }
        assert _resolve_diagram_type(spec, None) == "ObjectDiagram"

    def test_invalid_diagram_type_falls_back(self):
        spec = {"diagramType": "InvalidType", "classes": []}
        assert _resolve_diagram_type(spec, "ClassDiagram") == "ClassDiagram"


# ── Multi-diagram type conversions ───────────────────────────────────────────


def _mock_llm_state_machine(prompt: str) -> str:
    """Mock LLM that returns a state machine spec."""
    return json.dumps({
        "diagramType": "StateMachineDiagram",
        "states": [
            {"stateName": "Idle", "stateType": "initial"},
            {"stateName": "Processing", "stateType": "regular"},
            {"stateName": "Done", "stateType": "final"},
        ],
        "transitions": [
            {"source": "Idle", "target": "Processing", "trigger": "start"},
            {"source": "Processing", "target": "Done", "trigger": "complete"},
        ],
    })


def _mock_llm_object_diagram(prompt: str) -> str:
    """Mock LLM that returns an object diagram spec."""
    return json.dumps({
        "diagramType": "ObjectDiagram",
        "objects": [
            {"objectName": "alice", "className": "User", "attributes": [{"name": "name", "value": "Alice"}]},
            {"objectName": "order1", "className": "Order", "attributes": [{"name": "total", "value": "99.99"}]},
        ],
        "links": [
            {"source": "alice", "target": "order1", "relationshipType": "association"},
        ],
    })


def _mock_llm_agent_diagram(prompt: str) -> str:
    """Mock LLM that returns an agent diagram spec."""
    return json.dumps({
        "diagramType": "AgentDiagram",
        "intents": [
            {"intentName": "greet", "trainingPhrases": ["hello", "hi"]},
            {"intentName": "order", "trainingPhrases": ["I want to order"]},
        ],
        "states": [
            {"stateName": "welcome", "type": "state", "replies": [{"text": "Hello!", "replyType": "text"}]},
            {"stateName": "order_flow", "type": "state", "replies": [{"text": "What would you like?", "replyType": "text"}]},
        ],
        "transitions": [
            {"source": "initial", "target": "welcome", "condition": "when_intent_matched", "conditionValue": "greet"},
        ],
    })


class TestStateMachineConversion:
    """Test state machine diagram conversion."""

    def test_plantuml_state_machine(self):
        puml = "@startuml\n[*] --> Idle\nIdle --> Processing : start\nProcessing --> [*]\n@enduml"
        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64(puml),
            filename="states.puml",
            llm_predict=_mock_llm_state_machine,
        )
        assert result["action"] == "inject_complete_system"
        assert result["diagramType"] == "StateMachineDiagram"
        assert len(result["systemSpec"]["states"]) == 3
        assert len(result["systemSpec"]["transitions"]) == 2
        assert "state" in result["message"].lower()

    def test_generic_text_to_state_machine(self):
        text = "The order goes from Created to Processing when paid, then to Shipped, then Delivered."
        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64(text),
            filename="workflow.txt",
            llm_predict=_mock_llm_state_machine,
        )
        assert result["action"] == "inject_complete_system"
        assert result["diagramType"] == "StateMachineDiagram"

    def test_state_machine_empty_states_error(self):
        def mock_empty_states(prompt: str) -> str:
            return json.dumps({"diagramType": "StateMachineDiagram", "states": [], "transitions": []})

        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64("@startuml\n@enduml"),
            filename="empty.puml",
            llm_predict=mock_empty_states,
        )
        assert result["action"] == "agent_error"

    def test_state_machine_bad_transition_dropped(self):
        def mock_bad_trans(prompt: str) -> str:
            return json.dumps({
                "diagramType": "StateMachineDiagram",
                "states": [{"stateName": "Idle", "stateType": "initial"}],
                "transitions": [
                    {"source": "Idle", "target": "NonExistent", "trigger": "go"},
                ],
            })

        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64("@startuml\nstate Idle\n@enduml"),
            filename="sm.puml",
            llm_predict=mock_bad_trans,
        )
        assert result["action"] == "inject_complete_system"
        assert len(result["systemSpec"]["transitions"]) == 0


class TestObjectDiagramConversion:
    """Test object diagram conversion."""

    def test_object_diagram_conversion(self):
        text = "alice is a User with name=Alice. order1 is an Order with total=99.99."
        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64(text),
            filename="instances.txt",
            llm_predict=_mock_llm_object_diagram,
        )
        assert result["action"] == "inject_complete_system"
        assert result["diagramType"] == "ObjectDiagram"
        assert len(result["systemSpec"]["objects"]) == 2
        assert len(result["systemSpec"]["links"]) == 1
        assert "object" in result["message"].lower()

    def test_object_diagram_empty_objects_error(self):
        def mock_empty_objects(prompt: str) -> str:
            return json.dumps({"diagramType": "ObjectDiagram", "objects": [], "links": []})

        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64("nothing here"),
            filename="empty.txt",
            llm_predict=mock_empty_objects,
        )
        assert result["action"] == "agent_error"

    def test_object_diagram_invalid_link_dropped(self):
        def mock_bad_link(prompt: str) -> str:
            return json.dumps({
                "diagramType": "ObjectDiagram",
                "objects": [{"objectName": "a", "className": "A", "attributes": []}],
                "links": [{"source": "a", "target": "missing", "relationshipType": "assoc"}],
            })

        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64("object a : A"),
            filename="obj.txt",
            llm_predict=mock_bad_link,
        )
        assert result["action"] == "inject_complete_system"
        assert len(result["systemSpec"]["links"]) == 0


class TestAgentDiagramConversion:
    """Test agent diagram conversion."""

    def test_agent_diagram_conversion(self):
        text = "Bot has intents: greet, order. When greeting say Hello! When ordering ask What would you like?"
        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64(text),
            filename="chatbot.txt",
            llm_predict=_mock_llm_agent_diagram,
        )
        assert result["action"] == "inject_complete_system"
        assert result["diagramType"] == "AgentDiagram"
        assert len(result["systemSpec"]["intents"]) == 2
        assert len(result["systemSpec"]["states"]) == 2
        assert "intent" in result["message"].lower()

    def test_agent_diagram_empty_error(self):
        def mock_empty_agent(prompt: str) -> str:
            return json.dumps({"diagramType": "AgentDiagram", "intents": [], "states": [], "transitions": []})

        result = convert_file_to_diagram_spec(
            file_content_b64=_make_b64("nothing"),
            filename="bot.txt",
            llm_predict=mock_empty_agent,
        )
        assert result["action"] == "agent_error"


class TestBackwardCompatibility:
    """Ensure the old function alias still works."""

    def test_old_function_name_works(self):
        puml = "@startuml\nclass User {\n  +name: String\n}\n@enduml"
        result = convert_file_to_class_spec(
            file_content_b64=_make_b64(puml),
            filename="model.puml",
            llm_predict=_mock_llm_predict,
        )
        assert result["action"] == "inject_complete_system"
        assert result["diagramType"] == "ClassDiagram"

    def test_old_and_new_return_same_result(self):
        puml = "@startuml\nclass User {\n  +name: String\n}\n@enduml"
        b64 = _make_b64(puml)
        old = convert_file_to_class_spec(
            file_content_b64=b64,
            filename="model.puml",
            llm_predict=_mock_llm_predict,
        )
        new = convert_file_to_diagram_spec(
            file_content_b64=b64,
            filename="model.puml",
            llm_predict=_mock_llm_predict,
        )
        assert old == new
