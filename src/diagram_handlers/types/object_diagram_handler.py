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
from utilities.model_helpers import detailed_model_summary

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

        classes: Dict[str, Dict[str, Any]] = {}
        by_id: Dict[str, Dict[str, Any]] = {}

        for class_id, element in elements.items():
            if not isinstance(element, dict):
                continue
            if element.get("type") != "Class":
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
        return """You are a UML modeling expert. Create an object instance specification based on the user's request.

Return ONLY a JSON object with this structure:
{
  "objectName": "objectName",
  "className": "ClassName",
  "classId": "class_id_from_reference",
  "attributes": [
    {"name": "attributeName", "attributeId": "attr_id_from_reference", "value": "actualValue"}
  ]
}

CRITICAL RULES:
1. If a REFERENCE CLASS DIAGRAM is provided below, you MUST use ONLY the attributes from that diagram
2. DO NOT invent new attributes - use exactly what's defined in the reference class
3. Object name format: lowercase, e.g., "user1", "orderA"
4. ClassName and classId MUST match the reference diagram (if provided)
5. Each attribute MUST have:
   - name: EXACT attribute name from the class definition (just the name, without type or visibility)
   - attributeId: the EXACT id from the reference diagram
   - value: an ACTUAL example value (not a type)
6. Include ALL attributes from the referenced class with realistic example values
7. Keep values realistic and coherent
8. Do NOT include any "position" field - positioning is handled automatically
9. Return ONLY the JSON, no explanations

Examples:
- "create user object" -> {"objectName": "user1", "className": "User", "classId": "class_abc123", "attributes": [{"name": "id", "attributeId": "attr_xyz", "value": "001"}, {"name": "name", "attributeId": "attr_def", "value": "John Doe"}]}
- "create order object" -> {"objectName": "order1", "className": "Order", "classId": "class_ord456", "attributes": [{"name": "id", "attributeId": "attr_oid", "value": "ORD-001"}, {"name": "total", "attributeId": "attr_tot", "value": "99.99"}]}

Return ONLY the JSON, no explanations."""
    
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
            response = self.predict_with_retry(f"{system_prompt}\n\nUser Request: {user_prompt}")
            
            object_spec = self.parse_and_validate(
                response,
                required_keys=SINGLE_OBJECT_REQUIRED,
                optional_keys=SINGLE_OBJECT_OPTIONAL,
                label="ObjectDiagram.single_element",
            )
            
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
        system_prompt = """You are a UML modeling expert. Create a COMPLETE object diagram with multiple related object instances.

Return ONLY a JSON object with this structure:
{
  "systemName": "SystemName",
  "objects": [
    {
      "objectName": "object1",
      "className": "ClassName",
      "attributes": [
        {"name": "attr", "value": "actualValue"}
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
}

IMPORTANT RULES:
1. Create 3-6 related object instances
2. Each object should have 2-4 attributes with ACTUAL VALUES
3. Object names: lowercase (user1, order1, product2)
4. Include meaningful links between objects
5. Values should be realistic and coherent
6. Do NOT include any "position" field - positioning is handled automatically
7. Keep the scenario focused
8. If a REFERENCE CLASS DIAGRAM is provided, STRICTLY derive objects from it:
   - Use ONLY class names from the reference classes.
   - Every object MUST include className + classId from reference.
   - Every object attribute MUST include name + attributeId from reference.
   - Do NOT invent classes such as User/Order/Product unless they exist in the reference.
9. If the user asks "according to structural/class diagram", prioritise the reference model over generic examples.

Return ONLY the JSON, no explanations."""

        user_prompt = user_request
        if classes:
            user_prompt += "\n\nREFERENCE CLASS DIAGRAM (use these exact classes and attributes):\n"
            user_prompt += self._format_reference_classes(reference_diagram.get("elements", {}))
            user_prompt += "\n\nREFERENCE CLASS RELATIONSHIPS:\n"
            user_prompt += self._format_reference_relationships(class_relationships)

        try:
            response = self.predict_with_retry(f"{system_prompt}\n\nUser Request: {user_prompt}")
            
            system_spec = self.parse_and_validate(
                response,
                required_keys=SYSTEM_OBJECT_REQUIRED,
                optional_keys=SYSTEM_OBJECT_OPTIONAL,
                label="ObjectDiagram.complete_system",
            )

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
    
    def _format_reference_classes(self, elements: Dict[str, Any]) -> str:
        """Format reference diagram classes for LLM context"""
        formatted = []
        
        # Group elements by class
        classes = {k: v for k, v in elements.items() if v.get('type') == 'Class'}
        
        for class_id, class_data in classes.items():
            class_name = class_data.get('name', 'Unknown')
            formatted.append(f"\nClass: {class_name} (classId: {class_id})")
            formatted.append("Attributes:")
            
            # Get all attributes for this class
            for attr_id in class_data.get('attributes', []):
                if attr_id in elements:
                    attr = elements[attr_id]
                    attr_name = attr.get('name', '').replace('+ ', '').replace('- ', '').replace('# ', '')
                    # Extract just the attribute name (before the colon)
                    attr_name_only = attr_name.split(':')[0].strip()
                    formatted.append(f"  - {attr_name_only} (attributeId: {attr_id})")
        
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
        if reference_diagram and isinstance(reference_diagram, dict):
            ref_elements = reference_diagram.get("elements")
            if isinstance(ref_elements, dict):
                ref_classes = self._format_reference_classes(ref_elements)
                if ref_classes:
                    reference_context = (
                        "\n\nReference class diagram (use these classes and attributes "
                        "when creating or modifying objects):\n" + ref_classes
                    )

        system_prompt = """You are a UML modeling expert. The user wants to modify an existing object diagram.

Return ONLY a JSON object with one of these structures:

MODIFY OBJECT (rename or change class)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_object",
    "target": {
      "objectName": "currentObjectName"
    },
    "changes": {
      "objectName": "newObjectName"
    }
  }
}

ADD ATTRIBUTE VALUE (set or add attribute value on existing object)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_attribute_value",
    "target": {
      "objectName": "objectName",
      "attributeName": "attributeName"
    },
    "changes": {
      "value": "newValue"
    }
  }
}

ADD LINK (connect two objects)
{
  "action": "modify_model",
  "modification": {
    "action": "add_link",
    "target": {
      "sourceObject": "object1",
      "targetObject": "object2"
    },
    "changes": {
      "relationshipType": "association"
    }
  }
}

REMOVE ELEMENT (delete object or link)
{
  "action": "modify_model",
  "modification": {
    "action": "remove_element",
    "target": {
      "objectName": "objectToRemove"
    }
  }
}

IMPORTANT RULES:
1. Actions available: "modify_object", "modify_attribute_value", "add_link", "remove_element"
2. Always specify exact target names that exist in the current model
3. For remove_element, only specify the target — no "changes" needed
4. When the user asks for MULTIPLE changes at once (e.g., "set name and age on obj1"), use the "modifications" array format:
   { "action": "modify_model", "modifications": [ { "action": "...", "target": {...}, "changes": {...} }, ... ] }
5. Use "modification" (singular) for a single change, "modifications" (plural array) for multiple changes
6. Return ONLY the JSON object — no explanations or markdown

Return ONLY the JSON object — no explanations"""

        # Build context from current model using centralized helper
        context_block = ''
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, 'ObjectDiagram')
            if summary:
                context_block = f"\n\n{summary}"

        user_prompt = f"Modify the object diagram: {user_request}{context_block}{reference_context}"
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}"

        logger.info(f"[ObjectDiagram] generate_modification called with: {user_request!r}")

        try:
            response = self.predict_with_retry(full_prompt)
            json_text = self.clean_json_response(response)
            modification_spec = self.parse_json_safely(json_text)

            if not modification_spec:
                raise ValueError(f"Failed to parse modification JSON: {json_text[:300]}")

            self.validate_modification_spec(modification_spec)

            modification_spec.setdefault('action', 'modify_model')
            modification_spec.setdefault('diagramType', self.get_diagram_type())

            if 'message' not in modification_spec:
                if 'modifications' in modification_spec and isinstance(modification_spec['modifications'], list):
                    mods = modification_spec['modifications']
                    actions_summary = ", ".join(m.get('action', '?') for m in mods)
                    target_names = set()
                    for m in mods:
                        t = m.get('target', {})
                        n = t.get('objectName') or t.get('sourceObject') or 'element'
                        target_names.add(n)
                    modification_spec['message'] = f"Applied {len(mods)} modifications ({actions_summary}) to {', '.join(target_names)}"
                else:
                    mod_action = modification_spec['modification'].get('action', 'modification')
                    target = modification_spec['modification'].get('target', {})
                    target_name = target.get('objectName') or target.get('sourceObject') or 'element'
                    modification_spec['message'] = f"Applied {mod_action} to {target_name}"

            return modification_spec

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
