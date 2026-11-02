"""Every prompt in src/prompts/ is pinned by a hash of its rendered text.

A wording change is fine when it is meant: update the hash below in the same
commit (the failure prints the new one). What this catches is the unmeant
change: an edit that slips into a prompt while moving or refactoring code, or a
system prompt that stops being byte-identical (the provider's prompt cache
keys on an exact prefix). Constants are hashed as they are; functions are
rendered with the fixed sample values below.
"""
import ast
import hashlib
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import prompts

SAMPLES = {
    "user_request": "a library with books, members and loans",
    "request": "an AI support desk",
    "wanted": "- LoanStatus (referenced by Loan.status)",
    "request_text": "Loans are active, returned or overdue.",
    "class_summary": ["Book: title, isbn", "Member: name"],
    "rel_summary": ["Member -> Book (Association, 1..*)"],
    "algorithm_section": "KNOWN ALGORITHMS",
    "class_block": "\n\nCLASS BLOCK",
    "domain": "finance",
    "exemplars_block": "<!-- hero -->\n<section class='ds-hero'>X</section>",
    "interactions_block": "INTERACTIONS",
    "mockup_block": "\n\nMOCKUP",
    "class_metadata_text": "- Class \"Book\" (id: c1): title",
    "data_model_step": "1. DATA MODEL\n",
    "chart_rule": "CHART RULE.",
    "outline": "- Home: hero, table",
    "class_metadata": [{"id": "c1", "name": "Book", "attributes": [], "methods": []}],
    "has_classes": True,
    "image_count": 2,
    "user_message": "what can you do?",
    "message": "how do I model a loan?",
    "diagram_info": {"name": "Object Diagram", "description": "instances of classes"},
    "history_block": "User: hi\nAssistant: hello",
    "guide": "GUIDE",
    "full_summary": "ClassDiagram: Book, Member",
    "user_guide_block": "\n\nGUIDE BLOCK",
    "history_section": "HISTORY\n\n",
    "question": "what is a lifeline?",
    "context_summary": "CONTEXT",
    "intent_hint": "\nUser intent classified as: modeling_intent",
    "detected_targets_hint": "ClassDiagram",
    "generation_hint": "\nDetected generation request: web_app",
    "generator_prerequisites": "{\"web_app\": [\"ClassDiagram\"]}",
    "source_label": "PlantUML diagram",
    "target_diagram_type": "BPMN",
    "diagram_type": "AgentDiagram",
    "detect_mockup": True,
    "mockup_type": "GUINoCodeDiagram",
    "language": "French",
    "instructions": "Rails 7 with PostgreSQL.",
    "original": "A hotel app with rooms and bookings.",
    "spec": "Loans are active, returned or overdue.",
    "transcript": "user: hi\nassistant: hello",
}

EXPECTED = {
    "agent_diagram.AGENT_ACTIONS_BLOCK": "9576bf17624f1f60",
    "agent_diagram.AGENT_RULES_BLOCK": "29d8bb395a6eb76b",
    "agent_diagram.COMPLETE_SYSTEM_PROMPT": "e2d3955245930c68",
    "agent_diagram.MODIFY_SYSTEM_PROMPT_AGENT": "e29c7e7ff7a098a8",
    "agent_diagram.REPLY_TYPES_PROMPT": "8d27d04eb4f930fd",
    "agent_diagram.SYSTEM_PROMPT": "c83d635cc2a3df1f",
    "bpmn.AGENTIC_DESIGN_RULES": "6bfb87bfb31f89f9",
    "bpmn.AGENTIC_MODIFY_RULES": "ce0ccf0d925adf1f",
    "bpmn.AGENT_DIAGRAM_REF_RULE": "bde3b211af5dd496",
    "bpmn.MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN": "d3b48599be3377e1",
    "bpmn.MODIFY_SYSTEM_PROMPT_BPMN": "6115db1d6c30ee64",
    "bpmn.SYSTEM_PROMPT": "aa9a95a0a22a2c1d",
    "bpmn.agentic_reasoning_prompt": "e7cd6498e2bedfdf",
    "bpmn.reasoning_prompt": "3c3a57dbda476202",
    "class_diagram.CLASS_ACTIONS_BLOCK": "faab1c68556e7dfa",
    "class_diagram.CLASS_EXAMPLES_BLOCK": "6a50990625d2a4f5",
    "class_diagram.CLASS_KEY_RULES_BLOCK": "11eb400a5e0fa5b7",
    "class_diagram.ENUMERATION_RECOVERY_SYSTEM_PROMPT": "e97ad44fb611a1e0",
    "class_diagram.MODIFY_SYSTEM_PROMPT_CLASS": "4bd736670637974f",
    "class_diagram.SYSTEM_GENERATION_PROMPT": "75f201e43539a305",
    "class_diagram.SYSTEM_PROMPT": "70b5da254f7d69b1",
    "class_diagram.critique_prompt": "01686971a0e7ffbd",
    "class_diagram.enumeration_recovery_prompt": "448eb48877bbbbc8",
    "class_diagram.reasoning_prompt": "42f90fd9f5bb24be",
    "classifier.SYSTEM_PROMPT": "2e8c2a4188644e78",
    "component_diagram.MODIFY_SYSTEM_PROMPT_COMPONENT": "23991c77cac340e7",
    "component_diagram.SYSTEM_PROMPT": "5d7a169cb17017fd",
    "component_diagram.reasoning_prompt": "e72eb9a983f0ddb8",
    "conversation_memory.summary_prompt": "041305a671ac2b6a",
    "deployment_diagram.MODIFY_SYSTEM_PROMPT_DEPLOYMENT": "2a76df5aa7f12b4a",
    "deployment_diagram.SYSTEM_PROMPT": "c02291f851d758d0",
    "deployment_diagram.reasoning_prompt": "53784dd17449d526",
    "describe_model.history_section": "f980800abbebefec",
    "describe_model.qa_prompt": "d316e3ecf54f18c7",
    "describe_model.user_guide_block": "93891995a77c7844",
    "fallback.fallback_prompt": "dd14e0eb85cf0311",
    "file_conversion.AGENT_DIAGRAM_SCHEMA": "a8d3b3c7f6ef6186",
    "file_conversion.ALL_SCHEMAS": "88ba36a7bf1d5659",
    "file_conversion.BPMN_SCHEMA": "d1c16e3b099c8681",
    "file_conversion.CLASS_DIAGRAM_SCHEMA": "4ba7049ebd003736",
    "file_conversion.OBJECT_DIAGRAM_SCHEMA": "52796f60d791a274",
    "file_conversion.STATE_MACHINE_SCHEMA": "61a2534edf726b0a",
    "file_conversion.XMI_NOTE": "fda1d38660b4e15c",
    "file_conversion.auto_detect_image_prompt": "521fbab6b0ad6fe8",
    "file_conversion.auto_detect_prompt": "f4c9f1703dc981e4",
    "file_conversion.conversion_prompt": "505fc7660c096fe4",
    "file_conversion.diagram_type_label": "8d6aa53cd2c610c1",
    "file_conversion.image_prompt": "6bd3f1f4fc25ffa3",
    "fragments.ARCHITECTURE_MODIFY_SHARED_RULES": "86834cf1d9ba317c",
    "fragments.CHANGES_FIELD_RULE": "1eb461fcf1feea0f",
    "fragments.DELETE_CLASS_CASCADE_RULE": "12d5b78ad5cc3271",
    "fragments.ENUM_RULES_BLOCK": "8a1111b57bc4ab4b",
    "fragments.EXACT_NAMES_RULE": "59c83821f08f199f",
    "fragments.MODIFY_CRITICAL_BLOCK": "ce388cf75d234124",
    "fragments.MULTI_MOD_ARRAY_RULE": "13342726b4ec060c",
    "fragments.NAMING_PASCAL_RULE": "c0a47a840ca8b60d",
    "fragments.OCL_CONSTRAINT_BLOCK": "9d75d9f75f71a2b6",
    "fragments.OCL_EXAMPLES_BLOCK": "e9c348fe5a7ba4d3",
    "fragments.POSITION_DISCLAIMER": "8efeeaa50cad2558",
    "fragments.REMOVE_ELEMENT_RULE": "f696d66607dc6838",
    "fragments.RENAME_CASCADES_RULE": "fe0dcd858e01bdd7",
    "gui.CHART_RULE": "e5bac2afc7075015",
    "gui.CHART_RULE_FOR_CLASSES": "3253e11820b02b44",
    "gui.DATA_MODEL_STEP": "033f9d7024a9501b",
    "gui.MOCKUP_REASONING_ADDENDUM": "c2b3920b4a068408",
    "gui.SVG_THUMB_HINT": "d940d49684416059",
    "gui.complete_system_prompt": "45a7f77797f57a04",
    "gui.data_model_step_for_classes": "0133afd5b613e683",
    "gui.interactions_block": "42710b03fa350131",
    "gui.mockup_block": "f4152cebd1b87112",
    "gui.modification_prompt": "b7ade2f3c1936de6",
    "gui.reasoning_prompt": "9209345b96765cc7",
    "gui.system_prompt": "973ddb6fc8296eff",
    "model_operations.spec_block": "85d83868c96c8225",
    "modeling_help.bpmn_help_prompt": "f40ee7a7bf086b77",
    "modeling_help.component_help_prompt": "fb4bc14681792209",
    "modeling_help.deployment_help_prompt": "3e08456a533ed792",
    "modeling_help.generic_help_prompt": "230032dfddedacae",
    "modeling_help.quantum_help_prompt": "2c81dfe3f0686756",
    "modeling_help.user_profile_help_prompt": "1f5b530ff8ae2f3b",
    "object_diagram.COMPLETE_SYSTEM_PROMPT": "b7ae3192d97df2e1",
    "object_diagram.MODIFY_SYSTEM_PROMPT_OBJECT": "81362873673ae1d3",
    "object_diagram.SYSTEM_PROMPT": "56ae35fcebb20c2e",
    "planner.FEW_SHOT_EXAMPLES": "9c1e74d9a475399d",
    "planner.NON_WEB_APP_HINT": "befa401da2f4408e",
    "planner.planner_prompt": "18d25f2155d6194c",
    "quantum_circuit.ALGORITHM_TEMPLATES": "0e180461b0b5ccc5",
    "quantum_circuit.ALGORITHM_TEMPLATES_HEADER": "51ca332ee414e5a8",
    "quantum_circuit.MODIFY_SYSTEM_PROMPT": "1fd71139048d71d4",
    "quantum_circuit.SYSTEM_PROMPT": "7b749245192dac74",
    "quantum_circuit.complete_system_prompt": "05f176f69f2407fb",
    "reply_language.translation_prompt": "879d43f4fee00665",
    "spec_driven.framed_instructions": "a9d67d58a1379d49",
    "state_machine.MODIFY_SYSTEM_PROMPT_STATE_MACHINE": "df550a74a45076cd",
    "state_machine.SYSTEM_GENERATION_PROMPT": "acf22b03c16b0b65",
    "state_machine.SYSTEM_PROMPT": "1a9c0817932ef035",
    "state_machine.reasoning_prompt": "e5137fe7402ad8f2",
    "uml_rag.RAG_PROMPT": "04bd76f9d5ab9d35",
    "uml_rag.error_fallback_prompt": "39bbd9c99a1b51e1",
    "uml_rag.fallback_prompt": "f684f006348a7001",
    "user_profile.MODIFY_SYSTEM_PROMPT_USER": "1e5afd5ec6fcbca9",
    "user_profile.SYSTEM_GENERATION_PROMPT": "fcbc1da892e35284",
    "user_profile.SYSTEM_PROMPT": "55fdff2cf88681c0",
}


def _defined_names(module) -> list:
    """Top-level names the module itself assigns or defines (not imports)."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.FunctionDef):
            names.append(node.name)
    return [n for n in names if not n.startswith("_")]


def _rendered() -> dict:
    out = {}
    for info in pkgutil.iter_modules(prompts.__path__):
        module = importlib.import_module(f"prompts.{info.name}")
        for name in _defined_names(module):
            value = getattr(module, name)
            if callable(value):
                params = inspect.signature(value).parameters
                value = value(**{p: SAMPLES[p] for p in params})
            text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            out[f"{info.name}.{name}"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return out


def test_every_prompt_matches_its_pinned_hash():
    rendered = _rendered()
    changed = {k: v for k, v in rendered.items() if EXPECTED.get(k) != v}
    assert not changed, (
        "These prompts changed (or are new). If that was intended, pin the new hashes:\n"
        + "\n".join(f'    "{k}": "{v}",' for k, v in sorted(changed.items()))
    )
    assert set(EXPECTED) == set(rendered), f"pinned but gone: {sorted(set(EXPECTED) - set(rendered))}"

