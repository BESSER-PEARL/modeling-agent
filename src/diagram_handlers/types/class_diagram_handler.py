"""
Class Diagram Handler
Handles generation of UML Class Diagrams
"""

import logging
from typing import Dict, Any, List, Optional

from ..core.base_handler import (
    BaseDiagramHandler,
    LLMPredictionError,
    SINGLE_CLASS_REQUIRED,
    SINGLE_CLASS_OPTIONAL,
    SYSTEM_CLASS_REQUIRED,
    SYSTEM_CLASS_OPTIONAL,
)
from utilities.model_helpers import detailed_model_summary
from domain_patterns import get_pattern_hint

logger = logging.getLogger(__name__)


class ClassDiagramHandler(BaseDiagramHandler):
    """Handler for Class Diagram generation"""

    def get_diagram_type(self) -> str:
        return "ClassDiagram"

    def get_system_prompt(self) -> str:
        return """You are a UML modeling expert. Create a focused class specification based on the user's request.

Return ONLY a JSON object with this structure:
{
  "className": "ExactClassName",
  "attributes": [
    {"name": "attributeName", "type": "String", "visibility": "public"},
    {"name": "anotherAttr", "type": "int", "visibility": "private"}
  ],
  "methods": [
    {"name": "methodName", "returnType": "void", "visibility": "public", "parameters": [
      {"name": "paramName", "type": "String"}
    ]}
  ]
}

IMPORTANT RULES:
1. FOLLOW THE USER'S REQUEST STRICTLY - include exactly the attributes, methods, or details they specify
2. Create AS MANY attributes as needed (no fixed limits) based on what makes sense for the class
3. Methods: Generally SKIP methods unless the user asks for them. Only include a method if it's core to the domain logic (e.g., BankAccount.withdraw(), Order.calculateTotal()). Never include getters/setters.
4. If the user just says "create X class", generate relevant attributes and typically NO methods
5. Use proper programming conventions (camelCase for attributes/methods, PascalCase for classes)
6. visibility options: "public", "private", "protected", or "package" (default to "public")
7. Common types: String, int, boolean, double, Date, or custom class names
8. Method parameters are optional - empty array [] if no parameters needed
9. Do NOT include any "position" field - positioning is handled automatically
10. Return ONLY the JSON, no explanations or markdown

Examples:
- "create User class" -> attributes: id, username, email, password (4 attributes, 0-1 method)
- "create Product with inventory" -> attributes: id, name, price, stockQuantity, supplier (5+ attributes)
- "create BankAccount with deposit method" -> attributes: accountNumber, balance, owner + methods: deposit, withdraw

Return ONLY the JSON, no explanations."""

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a single class element with deterministic positioning."""

        system_prompt = self.get_system_prompt()
        user_prompt = f"Create a class specification for: {user_request}"

        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}"
        logger.info(f"[ClassDiagram] generate_single_element called with: {user_request!r}")
        logger.debug(f"[ClassDiagram] Full prompt length: {len(full_prompt)} chars")

        try:
            response = self.predict_with_retry(full_prompt)

            logger.info(f"[ClassDiagram] LLM raw response length: {len(response)}")
            logger.debug(f"[ClassDiagram] LLM raw response: {response[:500]!r}")

            simple_spec = self.parse_and_validate_with_repair(
                response,
                required_keys=SINGLE_CLASS_REQUIRED,
                optional_keys=SINGLE_CLASS_OPTIONAL,
                label="ClassDiagram.single_element",
            )

            # Remove any position the LLM might have hallucinated, then apply layout engine
            simple_spec.pop("position", None)
            self.apply_single_layout(simple_spec, existing_model)

            message = self._build_single_element_message(simple_spec)

            return {
                "action": "inject_element",
                "element": simple_spec,
                "diagramType": self.get_diagram_type(),
                "message": message
            }

        except LLMPredictionError as exc:
            logger.error(f"[ClassDiagram] generate_single_element LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't generate that class. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"[ClassDiagram] generate_single_element FAILED: {exc}", exc_info=True)
            return self.generate_fallback_element(user_request)

    def _get_system_generation_prompt(self) -> str:
        """Return the system prompt for complete class diagram generation."""
        return """You are a UML modeling expert. Create a COMPLETE, well-structured class diagram system.

Return ONLY a JSON object with this structure:
{
  "systemName": "SystemName",
  "classes": [
    {
      "className": "ClassName",
      "attributes": [
        {"name": "attr", "type": "String", "visibility": "public"}
      ],
      "methods": [
        {"name": "method", "returnType": "void", "visibility": "public", "parameters": [
          {"name": "param", "type": "String"}
        ]}
      ]
    }
  ],
  "relationships": [
    {
      "type": "Association",
      "source": "ClassName1",
      "target": "ClassName2",
      "sourceMultiplicity": "1",
      "targetMultiplicity": "*",
      "name": "relationshipName"
    }
  ]
}

IMPORTANT RULES:
1. FOLLOW THE USER'S REQUEST STRICTLY - include exactly the classes, attributes, methods, or relationships they specify
2. Create AS MANY classes as needed for a complete system (no fixed limits)
3. Each class should have AS MANY attributes as needed - don't artificially limit essential properties
4. Methods: Generally SKIP methods unless the user asks for them. Only include 1-2 methods per class MAX if they represent core domain behavior. Never include getters/setters.
5. Relationships are CRITICAL - always include meaningful connections:
   - "Association" - general relationship (most common)
   - "Inheritance" / "Generalization" - parent-child "is-a" (use sparingly)
   - "Composition" - strong "has-a" (part cannot exist without whole)
   - "Aggregation" - weak "has-a" (part can exist independently)
   - "Realization" - interface implementation
6. Relationship properties: "name", "sourceMultiplicity", "targetMultiplicity"
7. Use proper naming: PascalCase for classes, camelCase for attributes/methods
8. visibility: "public", "private", "protected", or "package"
9. Common types: String, int, boolean, double, Date, or custom class names
10. Do NOT include any "position" field - positioning is handled automatically
11. Return ONLY the JSON, no explanations or markdown

Examples:
- E-commerce system: User, Product, Order, Payment, ShoppingCart with appropriate associations
- Library system: Book, Author, Member, Loan with inheritance (DigitalBook extends Book) and compositions
- Banking system: Account, Customer, Transaction, Branch with aggregations and multiplicities

Return ONLY the JSON, no explanations."""

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a complete class diagram with two-pass reasoning, domain patterns,
        validation-feedback loop, and deterministic layout."""

        system_prompt = self._get_system_generation_prompt()

        # Inject domain pattern reference if the request matches a known domain
        pattern_hint = get_pattern_hint(user_request)
        if pattern_hint:
            system_prompt += pattern_hint

        logger.info(f"[ClassDiagram] generate_complete_system called with: {user_request!r}")
        logger.debug(f"[ClassDiagram] System prompt length: {len(system_prompt)} chars")

        try:
            # --- Two-pass generation: reason first, then produce JSON ---
            reasoning_prompt = (
                "You are a UML domain modeling expert. Think step by step about "
                "the following system request and plan the class diagram design.\n\n"
                f"User Request: {user_request}\n\n"
                "Analyze:\n"
                "1. What are the core domain entities (classes) needed?\n"
                "2. What attributes does each class need? (be thorough)\n"
                "3. What relationships connect these classes? What type (Association, "
                "Composition, Aggregation, Inheritance)? What multiplicities?\n"
                "4. Are there any association classes needed (e.g., Enrollment between "
                "Student and Course with grade)?\n"
                "5. Is there any inheritance hierarchy that makes sense?\n\n"
                "Provide a clear design analysis. Be thorough about relationships — "
                "they are the most commonly missed element."
            )

            response = self.predict_two_pass(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
            )

            logger.info(f"[ClassDiagram] System LLM response length: {len(response)}")
            logger.debug(f"[ClassDiagram] System LLM response: {response[:500]!r}")

            # Use parse_and_validate_with_repair for better error recovery
            system_spec = self.parse_and_validate_with_repair(
                response,
                required_keys=SYSTEM_CLASS_REQUIRED,
                optional_keys=SYSTEM_CLASS_OPTIONAL,
                label="ClassDiagram.complete_system",
            )

            logger.info(
                f"[ClassDiagram] Parsed system spec: "
                f"{len(system_spec.get('classes', []))} classes, "
                f"{len(system_spec.get('relationships', []))} relationships"
            )

            # --- Validation-feedback loop: self-critique and refine ---
            system_spec = self.validate_and_refine(
                system_spec,
                user_request=user_request,
                diagram_type="ClassDiagram",
            )

            # Strip any LLM-hallucinated positions, then apply deterministic layout
            for cls in system_spec.get("classes", []):
                cls.pop("position", None)
            self.apply_system_layout(system_spec, existing_model)

            message = self._build_system_message(system_spec)

            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": self.get_diagram_type(),
                "message": message
            }

        except LLMPredictionError as exc:
            logger.error(f"[ClassDiagram] generate_complete_system LLM FAILED: {exc}")
            # --- Graceful degradation: try generating classes one by one ---
            return self._incremental_system_fallback(user_request, existing_model)
        except Exception as exc:
            logger.error(f"[ClassDiagram] generate_complete_system FAILED: {exc}", exc_info=True)
            return self._incremental_system_fallback(user_request, existing_model)

    def _incremental_system_fallback(
        self, user_request: str, existing_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fallback: try to generate the system by creating classes individually.

        When the full system generation fails, this extracts class names from
        the user's request and generates each one separately, then combines
        them into a system spec.
        """
        logger.info("[ClassDiagram] Attempting incremental fallback generation")

        # Try to extract class names from the request
        extraction_prompt = (
            "From this request, extract ONLY the class/entity names the user wants. "
            "Return a JSON array of strings. Example: [\"User\", \"Product\", \"Order\"]\n\n"
            f"Request: {user_request}\n\n"
            "Return ONLY the JSON array, no explanations."
        )

        try:
            response = self.predict_with_retry(extraction_prompt, max_retries=1)
            cleaned = self.clean_json_response(response)
            import json as _json
            class_names = _json.loads(cleaned)
            if not isinstance(class_names, list) or len(class_names) == 0:
                raise ValueError("No class names extracted")
        except Exception:
            logger.warning("[ClassDiagram] Could not extract class names, using basic fallback")
            return self.generate_fallback_system()

        # Generate each class individually
        classes: List[Dict[str, Any]] = []
        for name in class_names[:10]:  # Cap at 10 to avoid excessive calls
            if not isinstance(name, str) or not name.strip():
                continue
            try:
                single_prompt = (
                    f"{self.get_system_prompt()}\n\n"
                    f"User Request: Create a {name} class with appropriate attributes "
                    f"for a system about: {user_request}"
                )
                resp = self.predict_with_retry(single_prompt, max_retries=1)
                spec = self.parse_and_validate(
                    resp,
                    required_keys=SINGLE_CLASS_REQUIRED,
                    optional_keys=SINGLE_CLASS_OPTIONAL,
                    label=f"ClassDiagram.incremental.{name}",
                )
                spec.pop("position", None)
                classes.append(spec)
                logger.info(f"[ClassDiagram] Incremental: generated class {name}")
            except Exception as exc:
                logger.warning(f"[ClassDiagram] Incremental: failed to generate {name}: {exc}")
                classes.append({
                    "className": name,
                    "attributes": [
                        {"name": "id", "type": "String", "visibility": "public"},
                    ],
                    "methods": [],
                })

        if not classes:
            return self.generate_fallback_system()

        system_spec = {
            "systemName": "System",
            "classes": classes,
            "relationships": [],
        }

        self.apply_system_layout(system_spec, existing_model)

        class_names_str = ", ".join(f"**{c.get('className', '?')}**" for c in classes)
        return {
            "action": "inject_complete_system",
            "systemSpec": system_spec,
            "diagramType": self.get_diagram_type(),
            "message": (
                f"I had some trouble generating the full system at once, but I created "
                f"{len(classes)} class(es): {class_names_str}. "
                "You may want to ask me to add relationships between them!"
            ),
        }

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        """Generate a fallback class when AI generation fails"""
        class_name = self.extract_name_from_request(request, "NewClass")

        fallback_spec = {
            "className": class_name,
            "attributes": [
                {"name": "id", "type": "String", "visibility": "public"},
                {"name": "name", "type": "String", "visibility": "private"}
            ],
            "methods": []
        }

        # Apply deterministic layout so the fallback doesn't render at 0,0
        self.apply_single_layout(fallback_spec)

        return {
            "action": "inject_element",
            "element": fallback_spec,
            "diagramType": self.get_diagram_type(),
            "message": f"I created a starter **{class_name}** class with some default attributes. Feel free to describe it in more detail and I'll refine it!"
        }

    def generate_fallback_system(self) -> Dict[str, Any]:
        """Generate a fallback system"""
        fallback_system = {
            "systemName": "BasicSystem",
            "classes": [
                {
                    "className": "Entity",
                    "attributes": [
                        {"name": "id", "type": "String", "visibility": "public"}
                    ],
                    "methods": []
                }
            ],
            "relationships": []
        }

        # Apply deterministic layout so the fallback doesn't render at 0,0
        self.apply_system_layout(fallback_system)

        return {
            "action": "inject_complete_system",
            "systemSpec": fallback_system,
            "diagramType": self.get_diagram_type(),
            "message": "I created a starter diagram with a basic **Entity** class. Describe your system in more detail (e.g. *'Create a library with books, authors, and members'*) and I'll build a richer model!"
        }
    
    # ------------------------------------------------------------------
    # Message Builders
    # ------------------------------------------------------------------

    def _build_single_element_message(self, spec: Dict[str, Any]) -> str:
        """Build a descriptive message for a single class creation."""
        name = spec.get("className", "Class")
        attrs = spec.get("attributes", [])
        methods = spec.get("methods", [])
        attr_names = [a.get("name", "") for a in attrs[:5]]
        parts = [f"Created the **{name}** class"]
        if attr_names:
            parts.append(f" with attributes: {', '.join(f'`{n}`' for n in attr_names)}")
            if len(attrs) > 5:
                parts.append(f" (+{len(attrs) - 5} more)")
        if methods:
            parts.append(f" and {len(methods)} method(s)")
        parts.append(". You can ask me to add relationships, new attributes, or create more classes!")
        return "".join(parts)

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        """Build a descriptive message for a complete class diagram system."""
        system_name = spec.get("systemName", "System")
        classes = spec.get("classes", [])
        rels = spec.get("relationships", [])
        class_names = [c.get("className", "?") for c in classes[:6]]
        msg = f"Built the **{system_name}** class diagram with {len(classes)} class(es)"
        if class_names:
            msg += f": {', '.join(f'**{n}**' for n in class_names)}"
            if len(classes) > 6:
                msg += f" (+{len(classes) - 6} more)"
        if rels:
            msg += f" and {len(rels)} relationship(s)"
        msg += ". Feel free to ask me to modify or extend any part of the diagram!"
        return msg

    # ------------------------------------------------------------------
    # Modification Support (Existing - Updated for new architecture)
    # ------------------------------------------------------------------
    
    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate modifications for existing class diagram elements.

        Enhanced with impact analysis: when renaming or removing a class,
        the LLM is informed of dependent relationships so it can cascade
        changes appropriately.
        """
        # Build impact context for modifications that affect relationships
        impact_context = self._build_impact_context(current_model)

        system_prompt = """You are a UML modeling expert. The user wants to modify an existing class diagram.

Return ONLY a JSON object with one of these structures:

MODIFY CLASS (rename or change properties)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_class",
    "target": {
      "className": "CurrentClassName"
    },
    "changes": {
      "name": "NewClassName"
    }
  }
}

ADD ATTRIBUTE (to existing class)
{
  "action": "modify_model",
  "modification": {
    "action": "add_attribute",
    "target": {
      "className": "ClassName"
    },
    "changes": {
      "name": "newAttribute",
      "type": "String",
      "visibility": "public"
    }
  }
}

MODIFY ATTRIBUTE (change existing attribute)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_attribute",
    "target": {
      "className": "ClassName",
      "attributeName": "oldAttributeName"
    },
    "changes": {
      "name": "newAttributeName",
      "type": "int",
      "visibility": "public"
    }
  }
}

ADD METHOD (to existing class)
{
  "action": "modify_model",
  "modification": {
    "action": "add_method",
    "target": {
      "className": "ClassName"
    },
    "changes": {
      "name": "newMethod",
      "returnType": "void",
      "visibility": "public",
      "parameters": [{"name": "param", "type": "String"}]
    }
  }
}

MODIFY METHOD (change existing method)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_method",
    "target": {
      "className": "ClassName",
      "methodName": "oldMethodName"
    },
    "changes": {
      "name": "newMethodName",
      "returnType": "boolean",
      "visibility": "public",
      "parameters": [{"name": "id", "type": "int"}]
    }
  }
}

ADD RELATIONSHIP (connect two classes)
{
  "action": "modify_model",
  "modification": {
    "action": "add_relationship",
    "target": {
      "sourceClass": "SourceClass",
      "targetClass": "TargetClass"
    },
    "changes": {
      "relationshipType": "Association",
      "sourceMultiplicity": "1",
      "targetMultiplicity": "*",
      "name": "relationshipName"
    }
  }
}

MODIFY RELATIONSHIP (change multiplicity, type, or name of an existing relationship)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_relationship",
    "target": {
      "sourceClass": "SourceClass",
      "targetClass": "TargetClass"
    },
    "changes": {
      "sourceMultiplicity": "1",
      "targetMultiplicity": "1..*"
    }
  }
}

REMOVE ELEMENT (delete class, attribute, method, or relationship)
{
  "action": "modify_model",
  "modification": {
    "action": "remove_element",
    "target": {
      "className": "ClassToRemove"
    }
  }
}

OR for removing attribute:
{
  "action": "modify_model",
  "modification": {
    "action": "remove_element",
    "target": {
      "className": "ClassName",
      "attributeName": "attributeToRemove"
    }
  }
}

MULTIPLE MODIFICATIONS (batch – use when the request needs more than one change)
{
  "action": "modify_model",
  "modifications": [
    {
      "action": "add_attribute",
      "target": { "className": "ClassName" },
      "changes": { "name": "attr1", "type": "String", "visibility": "public" }
    },
    {
      "action": "add_attribute",
      "target": { "className": "ClassName" },
      "changes": { "name": "attr2", "type": "int", "visibility": "private" }
    }
  ]
}

CASCADING CHANGES:
When renaming or removing a class, you MUST also update or remove any relationships
that reference that class. Use the "modifications" array to batch the class rename
AND all affected relationship updates in a single response.

IMPORTANT RULES:
1. Actions available: "modify_class", "add_attribute", "modify_attribute", "add_method", "modify_method", "add_relationship", "modify_relationship", "remove_element"
2. Always specify exact target names that exist in the current model
3. visibility options: "public", "private", "protected", "package"
4. Relationship types (case-sensitive): "Association", "Inheritance" (also called Generalization), "Composition", "Aggregation", "Realization"
5. Multiplicities: "1", "0..1", "*", "1..*", "0..*", or specific numbers like "5"
6. When adding methods, include empty parameters array [] if no parameters needed
7. When modifying, only include the fields that should change in "changes" object
8. For remove_element, only specify the target - no "changes" needed
9. Use "modify_relationship" (NOT "add_relationship") when the user wants to update/change an EXISTING relationship (e.g., change multiplicity, change type, rename)
10. Use "add_relationship" only when creating a brand-new connection between classes
11. When the user asks for MULTIPLE changes at once (e.g., "add several attributes", "add name and age to Person"), use the "modifications" array format with ALL changes in a single response
12. Use "modification" (singular) for a single change, "modifications" (plural array) for multiple changes
13. Return ONLY the JSON object – no explanations or markdown

Examples:
- "rename User class to Customer" -> modify_class with name change
- "add email attribute to User" -> add_attribute with type String, visibility private
- "make password private" -> modify_attribute changing visibility
- "add login method to User" -> add_method with appropriate returnType and parameters
- "connect Order to Customer" -> add_relationship with Association type
- "add generalization between Member and Author" -> add_relationship with Inheritance type (Member inherits from Author)
- "create inheritance from Student to Person" -> add_relationship with Inheritance type (Student is child, Person is parent)
- "change multiplicity to many" -> modify_relationship changing targetMultiplicity to "*"
- "Author should have several Childs, update the relation" -> modify_relationship with sourceClass Author, targetClass Childs, targetMultiplicity "1..*"
- "make the relation between Order and Product a composition" -> modify_relationship changing relationshipType to "Composition"
- "delete the temp attribute" -> remove_element with attributeName
- "add name, age, and email attributes to Person" -> use "modifications" array with 3 add_attribute entries
- "add several attributes to Book" -> use "modifications" array with multiple add_attribute entries (infer sensible attributes for the domain)

Return ONLY the JSON object – no explanations"""

        # Build context from current model using centralized helper
        context_block = ''
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, 'ClassDiagram')
            if summary:
                context_block = f"\n\n{summary}"

        # Add impact context (relationship dependencies per class)
        if impact_context:
            context_block += f"\n\n{impact_context}"

        user_prompt = f"Modify the class diagram: {user_request}{context_block}"
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}"

        logger.info(f"[ClassDiagram] generate_modification called with: {user_request!r}")
        logger.debug(f"[ClassDiagram] Modification context block length: {len(context_block)} chars")
        logger.debug(f"[ClassDiagram] Full modification prompt length: {len(full_prompt)} chars")

        try:
            response = self.predict_with_retry(full_prompt)

            logger.info(f"[ClassDiagram] Modification LLM response length: {len(response)}")
            logger.debug(f"[ClassDiagram] Modification LLM response: {response[:500]!r}")

            json_text = self.clean_json_response(response)
            modification_spec = self.parse_json_safely(json_text)
            
            if not modification_spec:
                raise ValueError(f"Failed to parse modification JSON: {json_text[:300]}")

            self.validate_modification_spec(modification_spec)

            modification_spec.setdefault('action', 'modify_model')
            modification_spec.setdefault('diagramType', self.get_diagram_type())

            if 'message' not in modification_spec:
                if 'modifications' in modification_spec and isinstance(modification_spec['modifications'], list):
                    modification_spec['message'] = self._friendly_batch_message(modification_spec['modifications'])
                elif 'modification' in modification_spec and isinstance(modification_spec['modification'], dict):
                    mod = modification_spec['modification']
                    act = mod.get('action', 'modification')
                    target = mod.get('target', {})
                    name = target.get('className') or target.get('attributeName') or target.get('methodName') or 'element'
                    modification_spec['message'] = self._friendly_mod_message(act, name)

            logger.info(
                f"[ClassDiagram] Modification spec: "
                f"batch={'modifications' in modification_spec}, "
                f"keys={list(modification_spec.keys())}"
            )
            
            return modification_spec
            
        except LLMPredictionError as exc:
            logger.error(f"[ClassDiagram] generate_modification LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't process that modification. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"[ClassDiagram] generate_modification FAILED: {exc}", exc_info=True)
            return self.generate_fallback_modification(user_request)
    
    def generate_fallback_modification(self, request: str) -> Dict[str, Any]:
        """Generate a fallback modification when AI generation fails"""
        return {
            "action": "modify_model",
            "modification": {
                "action": "modify_class",
                "target": {"className": "Unknown"},
                "changes": {"name": "ModifiedClass"}
            },
            "diagramType": self.get_diagram_type(),
            "message": "I couldn't apply that modification automatically. Could you rephrase your request? For example: *'Add a phone attribute to User'* or *'Create a relationship between Order and Product'*."
        }

    # ------------------------------------------------------------------
    # Impact Analysis Helpers
    # ------------------------------------------------------------------

    def _build_impact_context(self, model: Optional[Dict[str, Any]]) -> str:
        """Build a relationship dependency map for modification impact analysis.

        For each class, lists all relationships it participates in so the LLM
        knows to cascade changes when renaming or removing a class.
        """
        if not isinstance(model, dict):
            return ""

        elements = model.get("elements")
        relationships = model.get("relationships")
        if not isinstance(elements, dict) or not isinstance(relationships, dict):
            return ""

        # Build class ID -> name mapping
        class_names: Dict[str, str] = {}
        for eid, el in elements.items():
            if isinstance(el, dict) and el.get("type") == "Class":
                name = el.get("name", "")
                if name:
                    class_names[eid] = name

        if not class_names or not relationships:
            return ""

        # Build dependency map: class_name -> list of relationship descriptions
        deps: Dict[str, List[str]] = {name: [] for name in class_names.values()}
        for rel in relationships.values():
            if not isinstance(rel, dict):
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                continue
            src_id = source.get("element", "")
            tgt_id = target.get("element", "")
            src_name = class_names.get(src_id, "")
            tgt_name = class_names.get(tgt_id, "")
            rel_type = rel.get("type", "Association")
            if src_name and tgt_name:
                deps.setdefault(src_name, []).append(
                    f"{rel_type} -> {tgt_name}"
                )
                deps.setdefault(tgt_name, []).append(
                    f"{rel_type} <- {src_name}"
                )

        # Format as context block
        lines = ["Relationship dependencies (cascade changes if renaming/removing):"]
        for class_name, dep_list in deps.items():
            if dep_list:
                lines.append(f"  {class_name}: {', '.join(dep_list)}")

        return "\n".join(lines) if len(lines) > 1 else ""
