"""
Object Diagram Handler
Handles generation of UML Object Diagrams (instances of classes)
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from ..core.base_handler import (
    BaseDiagramHandler,
    LLMPredictionError,
    SINGLE_OBJECT_REQUIRED,
    SINGLE_OBJECT_OPTIONAL,
    SYSTEM_OBJECT_REQUIRED,
    SYSTEM_OBJECT_OPTIONAL,
)
from ..core.prompt_fragments import POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE
from schemas import SingleObjectSpec, SystemObjectSpec, ObjectModificationResponse
from utilities.model_context import detailed_model_summary

logger = logging.getLogger(__name__)


class ObjectDiagramHandler(BaseDiagramHandler):
    """Handler for Object Diagram generation"""
    
    def get_diagram_type(self) -> str:
        return "ObjectDiagram"

    def _sanitize_object_name(self, value: str, default_name: str = "object1") -> str:
        if not isinstance(value, str):
            return default_name
        base = re.sub(r"[^A-Za-z0-9_]", "", value.strip())
        if not base:
            return default_name
        if not base[0].isalpha():
            base = f"obj{base}"
        return base[0].lower() + base[1:]

    def _value_for_attribute(self, attr_name: str, attr_type: str, class_name: str, index: int) -> str:
        normalized_type = (attr_type or "").strip().lower()
        key = (attr_name or "").strip().lower()

        # Check if the attribute type is an enumeration — pick a valid literal
        enum_literals = getattr(self, '_enum_literals', {})
        if attr_type and attr_type in enum_literals and enum_literals[attr_type]:
            literals = enum_literals[attr_type]
            return literals[index % len(literals)]

        if "id" in key:
            prefix = re.sub(r"[^A-Za-z]", "", class_name).upper()[:3] or "OBJ"
            return f"{prefix}{index:03d}"
        if "name" in key:
            return f"{class_name}{index}"
        if "email" in key:
            return f"{class_name.lower()}{index}@example.com"
        if "date" in key:
            return "2026-01-01"
        if "time" in key:
            return "10:00:00"
        if "price" in key or "amount" in key or "cost" in key:
            return "99.99"
        if "count" in key or "quantity" in key or "copies" in key or "stock" in key:
            return "10"
        if "status" in key:
            return "active"

        if normalized_type in {"int", "integer", "long"}:
            return str(index)
        if normalized_type in {"float", "double", "decimal"}:
            return "1.0"
        if normalized_type in {"bool", "boolean"}:
            return "true"
        if normalized_type in {"date"}:
            return "2026-01-01"
        if normalized_type in {"time"}:
            return "10:00:00"
        if normalized_type in {"datetime", "timestamp"}:
            return "2026-01-01T10:00:00"
        return f"sample_{attr_name or 'value'}_{index}"

    def _extract_reference_catalog(
        self, reference_diagram: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
        if not isinstance(reference_diagram, dict):
            return {}, []

        elements = reference_diagram.get("elements")
        relationships = reference_diagram.get("relationships")
        if not isinstance(elements, dict):
            return {}, []

        # First pass: collect enumerations and their literals
        self._enum_literals: Dict[str, List[str]] = {}  # enum_name -> [literal_names]
        for el_id, element in elements.items():
            if not isinstance(element, dict):
                continue
            if element.get("type") != "Enumeration":
                continue
            enum_name = (element.get("name") or "").strip()
            if not enum_name:
                continue
            literals = []
            for attr_id in element.get("attributes", []):
                attr = elements.get(attr_id)
                if isinstance(attr, dict):
                    lit_name = (attr.get("name") or "").replace("+ ", "").replace("- ", "").replace("# ", "").split(":")[0].strip()
                    if lit_name:
                        literals.append(lit_name)
            self._enum_literals[enum_name] = literals

        classes: Dict[str, Dict[str, Any]] = {}
        by_id: Dict[str, Dict[str, Any]] = {}

        for class_id, element in elements.items():
            if not isinstance(element, dict):
                continue
            if element.get("type") not in ("Class", "AbstractClass"):
                continue
            class_name = element.get("name")
            if not isinstance(class_name, str) or not class_name.strip():
                continue
            class_name = class_name.strip()
            class_attrs: List[Dict[str, str]] = []
            for attr_id in element.get("attributes", []):
                if attr_id not in elements:
                    continue
                attr = elements.get(attr_id)
                if not isinstance(attr, dict):
                    continue
                if attr.get("type") != "ClassAttribute":
                    continue
                raw_name = attr.get("name", "")
                attr_name = str(raw_name).replace("+ ", "").replace("- ", "").replace("# ", "")
                attr_name = attr_name.split(":")[0].strip()
                if not attr_name:
                    continue
                class_attrs.append(
                    {
                        "name": attr_name,
                        "id": attr_id,
                        "type": str(attr.get("attributeType", "str")),
                    }
                )

            class_info = {
                "name": class_name,
                "id": class_id,
                "attributes": class_attrs,
            }
            classes[class_name.lower()] = class_info
            by_id[class_id] = class_info

        class_relationships: List[Dict[str, str]] = []
        if isinstance(relationships, dict):
            for relation in relationships.values():
                if not isinstance(relation, dict):
                    continue
                source = relation.get("source")
                target = relation.get("target")
                if not isinstance(source, dict) or not isinstance(target, dict):
                    continue
                source_element_id = source.get("element")
                target_element_id = target.get("element")
                if source_element_id not in by_id or target_element_id not in by_id:
                    continue
                rel_name = relation.get("name")
                if not isinstance(rel_name, str) or not rel_name.strip():
                    rel_name = "relatedTo"
                class_relationships.append(
                    {
                        "sourceClass": by_id[source_element_id]["name"],
                        "targetClass": by_id[target_element_id]["name"],
                        "name": rel_name.strip(),
                    }
                )

        return classes, class_relationships

    def _format_reference_relationships(self, relationships: List[Dict[str, str]]) -> str:
        if not relationships:
            return "No explicit class relationships were found."
        lines = []
        for rel in relationships:
            lines.append(
                f"- {rel['sourceClass']} -> {rel['targetClass']} (name: {rel['name']})"
            )
        return "\n".join(lines)

    def _build_reference_fallback_system(
        self,
        classes: Dict[str, Dict[str, Any]],
        relationships: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if not classes:
            return {
                "systemName": "BasicObjectDiagram",
                "objects": [],
                "links": [],
            }

        sorted_classes = sorted(classes.values(), key=lambda item: item["name"])[:6]
        objects: List[Dict[str, Any]] = []
        class_to_object: Dict[str, str] = {}

        for index, class_info in enumerate(sorted_classes, start=1):
            class_name = class_info["name"]
            object_name = self._sanitize_object_name(f"{class_name}{index}", f"object{index}")
            class_to_object[class_name.lower()] = object_name
            attributes = []
            for attr in class_info.get("attributes", []):
                attributes.append(
                    {
                        "name": attr["name"],
                        "attributeId": attr["id"],
                        "value": self._value_for_attribute(
                            attr["name"], attr.get("type", "str"), class_name, index
                        ),
                    }
                )

            objects.append(
                {
                    "objectName": object_name,
                    "className": class_name,
                    "classId": class_info["id"],
                    "attributes": attributes,
                }
            )

        links: List[Dict[str, str]] = []
        for relation in relationships:
            source_obj = class_to_object.get(relation["sourceClass"].lower())
            target_obj = class_to_object.get(relation["targetClass"].lower())
            if not source_obj or not target_obj:
                continue
            links.append(
                {
                    "source": source_obj,
                    "target": target_obj,
                    "relationshipType": relation["name"],
                }
            )

        return {
            "systemName": "ObjectDiagramFromStructuralModel",
            "objects": objects,
            "links": links,
        }

    def _normalize_system_from_reference(
        self,
        system_spec: Dict[str, Any],
        classes: Dict[str, Dict[str, Any]],
        relationships: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if not isinstance(system_spec, dict):
            return self._build_reference_fallback_system(classes, relationships)

        raw_objects = system_spec.get("objects")
        if not isinstance(raw_objects, list):
            return self._build_reference_fallback_system(classes, relationships)

        normalized_objects: List[Dict[str, Any]] = []
        object_lookup: Dict[str, Dict[str, str]] = {}
        per_class_counter: Dict[str, int] = {}

        for raw_obj in raw_objects:
            if not isinstance(raw_obj, dict):
                continue

            class_name_raw = raw_obj.get("className")
            if not isinstance(class_name_raw, str) or not class_name_raw.strip():
                continue
            class_info = classes.get(class_name_raw.strip().lower())
            if not class_info:
                continue

            class_name = class_info["name"]
            per_class_counter[class_name] = per_class_counter.get(class_name, 0) + 1
            object_index = per_class_counter[class_name]
            fallback_object_name = f"{class_name}{object_index}"
            object_name = self._sanitize_object_name(
                str(raw_obj.get("objectName", fallback_object_name)),
                default_name=fallback_object_name[0].lower() + fallback_object_name[1:],
            )

            raw_attrs = raw_obj.get("attributes") if isinstance(raw_obj.get("attributes"), list) else []
            incoming_by_name: Dict[str, Dict[str, Any]] = {}
            for attr in raw_attrs:
                if not isinstance(attr, dict):
                    continue
                attr_name = attr.get("name")
                if not isinstance(attr_name, str):
                    continue
                incoming_by_name[attr_name.strip().lower()] = attr

            normalized_attrs: List[Dict[str, str]] = []
            for ref_attr in class_info.get("attributes", []):
                ref_attr_name = ref_attr["name"]
                incoming_attr = incoming_by_name.get(ref_attr_name.lower(), {})
                value = incoming_attr.get("value")
                if not isinstance(value, str) or not value.strip():
                    value = self._value_for_attribute(
                        ref_attr_name, ref_attr.get("type", "str"), class_name, object_index
                    )
                normalized_attrs.append(
                    {
                        "name": ref_attr_name,
                        "attributeId": ref_attr["id"],
                        "value": value,
                    }
                )

            normalized_obj = {
                "objectName": object_name,
                "className": class_name,
                "classId": class_info["id"],
                "attributes": normalized_attrs,
            }
            normalized_objects.append(normalized_obj)
            object_lookup[object_name.lower()] = {
                "className": class_name,
                "objectName": object_name,
            }

        if not normalized_objects:
            return self._build_reference_fallback_system(classes, relationships)

        known_class_pairs = {
            (rel["sourceClass"].lower(), rel["targetClass"].lower()): rel["name"]
            for rel in relationships
        }
        known_class_pairs.update(
            {
                (rel["targetClass"].lower(), rel["sourceClass"].lower()): rel["name"]
                for rel in relationships
            }
        )

        normalized_links: List[Dict[str, str]] = []
        raw_links = system_spec.get("links") if isinstance(system_spec.get("links"), list) else []
        for raw_link in raw_links:
            if not isinstance(raw_link, dict):
                continue
            source_name = raw_link.get("source")
            target_name = raw_link.get("target")
            if not isinstance(source_name, str) or not isinstance(target_name, str):
                continue
            source_obj = object_lookup.get(source_name.strip().lower())
            target_obj = object_lookup.get(target_name.strip().lower())
            if not source_obj or not target_obj:
                continue

            rel_name = raw_link.get("relationshipType")
            if not isinstance(rel_name, str) or not rel_name.strip():
                rel_name = known_class_pairs.get(
                    (
                        source_obj["className"].lower(),
                        target_obj["className"].lower(),
                    ),
                    "relatedTo",
                )

            normalized_links.append(
                {
                    "source": source_obj["objectName"],
                    "target": target_obj["objectName"],
                    "relationshipType": rel_name.strip(),
                }
            )

        if not normalized_links and relationships:
            first_obj_by_class: Dict[str, str] = {}
            for obj in normalized_objects:
                class_key = obj["className"].lower()
                if class_key not in first_obj_by_class:
                    first_obj_by_class[class_key] = obj["objectName"]

            for rel in relationships:
                source_obj = first_obj_by_class.get(rel["sourceClass"].lower())
                target_obj = first_obj_by_class.get(rel["targetClass"].lower())
                if not source_obj or not target_obj:
                    continue
                normalized_links.append(
                    {
                        "source": source_obj,
                        "target": target_obj,
                        "relationshipType": rel["name"],
                    }
                )

        system_name = system_spec.get("systemName")
        if not isinstance(system_name, str) or not system_name.strip():
            system_name = "ObjectDiagramFromStructuralModel"

        return {
            "systemName": system_name.strip(),
            "objects": normalized_objects,
            "links": normalized_links,
        }
    
    def get_system_prompt(self) -> str:
        return f"""You are a UML modeling expert. Create an object instance specification based on the user's request.

CRITICAL RULES:
1. If a REFERENCE CLASS DIAGRAM is provided below, you MUST use ONLY the attributes from that diagram
2. DO NOT invent new attributes - use exactly what's defined in the reference class
3. objectName must be a short lowercase instance identifier like "user1", "orderA", "task2". NEVER include the class name or a colon in objectName — just the instance identifier.
4. ClassName and classId MUST match the reference diagram (if provided)
5. ENUMERATIONS — STRICT RULE: If an attribute's type matches the name of an enumeration listed in the reference, the value MUST be one of that enumeration's valid literals (shown as "valid values: ..."). NEVER invent enum values like "Fiction" or "Active" if they are not in the listed literals. Pick the closest matching literal from the list.
6. Each attribute MUST have:
   - name: EXACT attribute name from the class definition (just the name, without type or visibility)
   - attributeId: the EXACT id from the reference diagram
   - value: an ACTUAL example value (not a type). For enum-typed attributes, this MUST be one of the valid literals from rule #5.
7. Include ALL attributes from the referenced class with realistic example values
8. Keep values realistic and coherent
9. {POSITION_DISCLAIMER}"""
    
    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None,
                                reference_diagram: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a single object instance with deterministic positioning."""
        
        system_prompt = self.get_system_prompt()
        
        # Build user prompt with reference diagram context
        user_prompt = f"Create an object specification for: {user_request}"
        
        if reference_diagram and reference_diagram.get('elements'):
            user_prompt += "\n\nREFERENCE CLASS DIAGRAM (use these exact class and attribute definitions):\n"
            user_prompt += self._format_reference_classes(reference_diagram['elements'])
        
        try:
            parsed = self.predict_structured(user_prompt, SingleObjectSpec, system_prompt=system_prompt)
            object_spec = parsed.model_dump()

            # Sanitize objectName: strip any ": ClassName" suffix the LLM may have included
            raw_name = object_spec.get("objectName", "")
            class_name = object_spec.get("className", "")
            if class_name and ":" in raw_name:
                raw_name = raw_name.split(":")[0].strip()
            object_spec["objectName"] = self._sanitize_object_name(raw_name, default_name=f"{class_name[0].lower()}{class_name[1:]}1" if class_name else "object1")

            # Enrich each attribute with its type from the reference class diagram so the
            # frontend ObjectAttribute element can carry the correct attributeType
            # (otherwise it defaults to 'str' and breaks enum/int/date/float typing).
            self._enrich_attribute_types(object_spec, reference_diagram)

            # Remove any hallucinated position and apply deterministic layout
            object_spec.pop("position", None)
            self.apply_single_layout(object_spec, existing_model)
            
            return {
                "action": "inject_element",
                "element": object_spec,
                "diagramType": "ObjectDiagram",
                "message": self._build_single_object_message(object_spec)
            }
            
        except LLMPredictionError:
            logger.error("[ObjectDiagram] generate_single_element LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that object. Please try again or rephrase your request.")
        except Exception:
            logger.error("[ObjectDiagram] generate_single_element FAILED", exc_info=True)
            return self.generate_fallback_element(user_request)
    
    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None,
                                reference_diagram: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a complete object diagram with deterministic positioning."""

        classes, class_relationships = self._extract_reference_catalog(reference_diagram)
        system_prompt = f"""You are a UML modeling expert. Create a COMPLETE object diagram with multiple related object instances.

Before generating, think through:
- What object instances best illustrate this scenario?
- What realistic attribute values make the example coherent?
- What links between objects reflect the class diagram relationships?
- If a reference class diagram is provided, which classes should be instantiated?

IMPORTANT RULES:
1. Create 3-6 related object instances
2. Each object should have 2-4 attributes with ACTUAL VALUES
3. Object names: lowercase instance name + number (user1, order1, product2). NEVER use the class name as the object name — "Order: Order" is WRONG, use "order1: Order".
4. Include meaningful links between objects
5. Values should be realistic and coherent
6. {POSITION_DISCLAIMER}
7. Keep the scenario focused
8. If a REFERENCE CLASS DIAGRAM is provided, STRICTLY derive objects from it:
   - Use ONLY class names from the reference classes.
   - Every object MUST include className + classId from reference.
   - Every object attribute MUST include name + attributeId from reference.
   - Do NOT invent classes such as User/Order/Product unless they exist in the reference.
9. If the user asks "according to structural/class diagram", prioritise the reference model over generic examples."""

        user_prompt = user_request
        if classes:
            user_prompt += "\n\nREFERENCE CLASS DIAGRAM (use these exact classes and attributes):\n"
            user_prompt += self._format_reference_classes(reference_diagram.get("elements", {}))
            user_prompt += "\n\nREFERENCE CLASS RELATIONSHIPS:\n"
            user_prompt += self._format_reference_relationships(class_relationships)

        try:
            parsed = self.predict_structured(
                user_prompt, SystemObjectSpec, system_prompt=system_prompt
            )
            system_spec = parsed.model_dump()

            if classes:
                system_spec = self._normalize_system_from_reference(
                    system_spec, classes, class_relationships
                )
            
            # Strip any hallucinated positions and apply deterministic layout
            for obj in system_spec.get("objects", []):
                obj.pop("position", None)
            self.apply_system_layout(system_spec, existing_model)
            
            mode_note = " from structural model" if classes else ""
            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": "ObjectDiagram",
                "message": self._build_object_system_message(system_spec, mode_note)
            }
            
        except LLMPredictionError:
            logger.error("[ObjectDiagram] generate_complete_system LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that object diagram. Please try again or rephrase your request.")
        except Exception:
            logger.error("[ObjectDiagram] generate_complete_system FAILED", exc_info=True)
            return self.generate_fallback_system()
    
    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        """Generate a fallback object when AI generation fails"""
        object_name = self.extract_name_from_request(request, "object1").lower()
        class_name = self.extract_name_from_request(request, "Entity")
        
        fallback_spec = {
            "objectName": object_name,
            "className": class_name,
            "attributes": [
                {"name": "id", "value": "001"},
                {"name": "name", "value": "Sample"}
            ]
        }

        # Apply deterministic layout so the fallback doesn't render at 0,0
        self.apply_single_layout(fallback_spec)
        
        return {
            "action": "inject_element",
            "element": fallback_spec,
            "diagramType": "ObjectDiagram",
            "message": f"I created a starter **{object_name}** object (instance of {class_name}). Describe the scenario in more detail (e.g. which class it represents and its attribute values) for a more accurate result!"
        }
    
    def generate_fallback_system(self) -> Dict[str, Any]:
        """Generate a fallback object diagram"""
        fallback_system = {
            "systemName": "BasicObjectDiagram",
            "objects": [
                {
                    "objectName": "instance1",
                    "className": "Entity",
                    "attributes": [
                        {"name": "id", "value": "001"}
                    ]
                }
            ],
            "links": []
        }

        # Apply deterministic layout so the fallback doesn't render at 0,0
        self.apply_system_layout(fallback_system)

        return {
            "action": "inject_complete_system",
            "systemSpec": fallback_system,
            "diagramType": "ObjectDiagram",
            "message": "I created a starter object diagram. Describe your scenario in more detail (e.g. *'Create objects for a library with 2 books and 1 author'*) and I'll build a richer diagram!"
        }
    
    def _enrich_attribute_types(self, object_spec: Dict[str, Any], reference_diagram: Optional[Dict[str, Any]]) -> None:
        """Populate each attribute's 'type' field from the reference class diagram.

        The LLM only returns name/value/attributeId. The frontend needs the
        attribute type so it can render the ObjectAttribute element with the
        correct attributeType (instead of defaulting to 'str').
        """
        if not isinstance(reference_diagram, dict):
            return
        elements = reference_diagram.get('elements')
        if not isinstance(elements, dict):
            return

        class_name = object_spec.get('className', '')
        class_id = object_spec.get('classId')

        # Build a name -> attribute lookup for the target class as a fallback
        # when the LLM omitted attributeId.
        target_class = None
        if class_id and class_id in elements:
            target_class = elements[class_id]
        else:
            for el in elements.values():
                if isinstance(el, dict) and el.get('type') in ('Class', 'AbstractClass') and el.get('name') == class_name:
                    target_class = el
                    break

        attr_name_to_type: Dict[str, str] = {}
        attr_name_to_id: Dict[str, str] = {}
        if target_class:
            for attr_id in target_class.get('attributes', []):
                attr_el = elements.get(attr_id)
                if isinstance(attr_el, dict):
                    raw_name = (attr_el.get('name') or '').replace('+ ', '').replace('- ', '').replace('# ', '').split(':')[0].strip()
                    attr_type = attr_el.get('attributeType', 'str')
                    if raw_name:
                        attr_name_to_type[raw_name] = attr_type
                        attr_name_to_id[raw_name] = attr_id

        for attr in object_spec.get('attributes', []) or []:
            if not isinstance(attr, dict):
                continue
            attr_type = None
            attr_id = attr.get('attributeId')
            if attr_id and attr_id in elements and isinstance(elements[attr_id], dict):
                attr_type = elements[attr_id].get('attributeType')
            if not attr_type:
                attr_type = attr_name_to_type.get(attr.get('name', ''))
            if not attr.get('attributeId'):
                attr['attributeId'] = attr_name_to_id.get(attr.get('name', ''))
            attr['type'] = attr_type or 'str'

    def _format_reference_classes(self, elements: Dict[str, Any]) -> str:
        """Format reference diagram classes for LLM context"""
        formatted = []

        # Extract enumeration literals on the fly (don't rely on _extract_reference_catalog
        # being called first — generate_single_element calls this method directly).
        enum_literals: Dict[str, List[str]] = {}
        for el in elements.values():
            if not isinstance(el, dict) or el.get('type') != 'Enumeration':
                continue
            enum_name = (el.get('name') or '').strip()
            if not enum_name:
                continue
            literals: List[str] = []
            for attr_id in el.get('attributes', []):
                attr = elements.get(attr_id)
                if isinstance(attr, dict):
                    lit_name = (attr.get('name') or '').replace('+ ', '').replace('- ', '').replace('# ', '').split(':')[0].strip()
                    if lit_name:
                        literals.append(lit_name)
            enum_literals[enum_name] = literals

        # Cache for any subsequent code path that reads self._enum_literals
        self._enum_literals = enum_literals

        # List enumerations and their valid values
        for enum_name, literals in enum_literals.items():
            if literals:
                formatted.append(f"\nEnumeration: {enum_name} — valid values: {', '.join(literals)}")

        # Group elements by class
        classes = {k: v for k, v in elements.items() if v.get('type') in ('Class', 'AbstractClass')}

        for class_id, class_data in classes.items():
            class_name = class_data.get('name', 'Unknown')
            formatted.append(f"\nClass: {class_name} (classId: {class_id})")
            formatted.append("Attributes:")

            # Get all attributes for this class
            for attr_id in class_data.get('attributes', []):
                if attr_id in elements:
                    attr = elements[attr_id]
                    attr_name = attr.get('name', '').replace('+ ', '').replace('- ', '').replace('# ', '')
                    attr_name_only = attr_name.split(':')[0].strip()
                    attr_type = attr.get('attributeType', 'str')
                    type_info = f", type: {attr_type}" if attr_type else ""
                    # If this is an enum type, show valid values
                    if attr_type in enum_literals:
                        type_info += f" [valid values: {', '.join(enum_literals[attr_type])}]"
                    formatted.append(f"  - {attr_name_only} (attributeId: {attr_id}{type_info})")

        return '\n'.join(formatted)

    # ------------------------------------------------------------------
    # Message Builders
    # ------------------------------------------------------------------

    def _build_single_object_message(self, spec: Dict[str, Any]) -> str:
        """Build a descriptive message for a single object creation."""
        obj_name = spec.get("objectName", "object")
        cls_name = spec.get("className", "Class")
        attrs = spec.get("attributes", [])
        msg = f"Created **{obj_name}** (an instance of **{cls_name}**)"
        if attrs:
            preview = [f'`{a.get("name", "")}={a.get("value", "")}`' for a in attrs[:4]]
            msg += f" with values: {', '.join(preview)}"
            if len(attrs) > 4:
                msg += f" (+{len(attrs) - 4} more)"
        msg += ". You can ask me to add more objects or links between them!"
        return msg

    def _build_object_system_message(self, spec: Dict[str, Any], mode_note: str = "") -> str:
        """Build a descriptive message for a complete object diagram."""
        system_name = spec.get("systemName", "ObjectDiagram")
        objects = spec.get("objects", [])
        links = spec.get("links", [])
        obj_names = [o.get("objectName", "?") for o in objects[:6]]
        msg = f"Built the **{system_name}** object diagram{mode_note} with {len(objects)} object(s)"
        if obj_names:
            msg += f": {', '.join(f'**{n}**' for n in obj_names)}"
            if len(objects) > 6:
                msg += f" (+{len(objects) - 6} more)"
        if links:
            msg += f" and {len(links)} link(s)"
        msg += ". Feel free to ask me to modify values or add more objects!"
        return msg

    # ------------------------------------------------------------------
    # Modification Support
    # ------------------------------------------------------------------

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate modifications for existing object diagram elements."""

        # If a reference class diagram is available, include it for context
        reference_diagram = kwargs.get("reference_diagram")
        reference_context = ""
        reference_classes: Dict[str, Dict[str, Any]] = {}
        if reference_diagram and isinstance(reference_diagram, dict):
            ref_elements = reference_diagram.get("elements")
            if isinstance(ref_elements, dict):
                ref_classes = self._format_reference_classes(ref_elements)
                if ref_classes:
                    reference_context = (
                        "\n\nReference class diagram (use these classes and attributes "
                        "when creating or modifying objects):\n" + ref_classes
                    )
                # Catalog for deterministic server-side resolution of classId /
                # attributeId — the LLM often omits them even when instructed.
                reference_classes, _ = self._extract_reference_catalog(reference_diagram)

        system_prompt = (
            """You are a UML modeling expert. The user wants to modify an object diagram.

IMPORTANT RULES:
1. Actions available: "add_object", "modify_object", "modify_attribute_value", "add_link", "remove_element"
2. add_object: set target.objectName to a short lowercase instance identifier (e.g. "user2", "order1"). NEVER include the class name or a colon in objectName. Put className, classId, and attributes (with concrete values) in "changes".
3. For existing elements, always specify exact target names from the current model
4. """
            + REMOVE_ELEMENT_RULE
            + """
5. When the user asks for MULTIPLE changes at once, return multiple entries in the modifications array
6. If a reference class diagram is provided, you MUST copy its classId and per-attribute attributeId verbatim into the modification — these ids are what the frontend uses to link the object back to its class definition.
7. Example: "add an object user2 of class User" → add_object with target.objectName="user2", changes.className="User", changes.classId="<id from reference>", changes.attributes=[{name:"id",attributeId:"<id>",value:"USR002"},{name:"name",attributeId:"<id>",value:"Bob"}]"""
        )

        # Build context from current model using centralized helper
        context_block = ''
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, 'ObjectDiagram')
            if summary:
                context_block = f"\n\n{summary}"

        user_prompt = f"Modify the object diagram: {user_request}{context_block}{reference_context}"

        logger.info(f"[ObjectDiagram] generate_modification called with: {user_request!r}")

        def _resolve_class_references(mod_list: list) -> list:
            """Fill in classId / attributeId for add_object mods from the reference catalog.

            The LLM routinely forgets these ids even when the prompt demands them.
            Since we already loaded the reference catalog above, resolve them
            deterministically so the frontend can link the new object to its
            class definition without needing a round-trip.
            """
            if not reference_classes:
                return mod_list

            for mod in mod_list:
                if not isinstance(mod, dict) or mod.get("action") != "add_object":
                    continue
                changes = mod.get("changes") or {}
                class_name = (changes.get("className") or "").strip()
                if not class_name:
                    continue
                class_info = reference_classes.get(class_name.lower())
                if not class_info:
                    continue

                # Canonicalise the class name to the exact casing from the reference
                changes["className"] = class_info["name"]
                if not changes.get("classId"):
                    changes["classId"] = class_info["id"]

                # Resolve attributeId per attribute by matching name (case-insensitive).
                attr_by_name = {
                    a["name"].lower(): a for a in class_info.get("attributes", [])
                }
                for attr in changes.get("attributes") or []:
                    if not isinstance(attr, dict):
                        continue
                    ref_attr = attr_by_name.get((attr.get("name") or "").strip().lower())
                    if not ref_attr:
                        continue
                    if not attr.get("attributeId"):
                        attr["attributeId"] = ref_attr["id"]
                    if not attr.get("type"):
                        attr["type"] = ref_attr.get("type", "str")

                mod["changes"] = changes
            return mod_list

        try:
            return self._execute_modification(
                user_prompt, system_prompt, ObjectModificationResponse,
                post_processor=_resolve_class_references,
            )

        except LLMPredictionError as exc:
            logger.error(f"[ObjectDiagram] generate_modification LLM FAILED: {exc}")
            return self._error_response("I couldn't process that modification. Please try again or rephrase your request.")
        except Exception as exc:
            logger.error(f"[ObjectDiagram] generate_modification FAILED: {exc}", exc_info=True)
            return {
                "action": "modify_model",
                "modification": {
                    "action": "modify_object",
                    "target": {"objectName": "Unknown"},
                    "changes": {"objectName": "ModifiedObject"}
                },
                "diagramType": self.get_diagram_type(),
                "message": "I couldn't apply that modification automatically. Could you rephrase it? For example: *'Change the name of object X to Y'* or *'Add a link between obj1 and obj2'*."
            }
