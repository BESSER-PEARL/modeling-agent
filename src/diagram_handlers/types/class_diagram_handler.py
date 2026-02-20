"""
Class Diagram Handler
Handles generation of UML Class Diagrams
"""

import logging
from typing import Dict, Any

from ..core.base_handler import (
    BaseDiagramHandler,
    SINGLE_CLASS_REQUIRED,
    SINGLE_CLASS_OPTIONAL,
    SYSTEM_CLASS_REQUIRED,
    SYSTEM_CLASS_OPTIONAL,
)
from utilities.model_helpers import detailed_model_summary

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

            simple_spec = self.parse_and_validate(
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

        except Exception as exc:
            logger.error(f"[ClassDiagram] generate_single_element FAILED: {exc}", exc_info=True)
            return self.generate_fallback_element(user_request)

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a complete class diagram with multiple classes and deterministic layout."""

        system_prompt = """You are a UML modeling expert. Create a COMPLETE, well-structured class diagram system.

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

        full_prompt = f"{system_prompt}\n\nUser Request: {user_request}"
        logger.info(f"[ClassDiagram] generate_complete_system called with: {user_request!r}")
        logger.debug(f"[ClassDiagram] System prompt length: {len(full_prompt)} chars")

        try:
            response = self.predict_with_retry(full_prompt)

            logger.info(f"[ClassDiagram] System LLM response length: {len(response)}")
            logger.debug(f"[ClassDiagram] System LLM response: {response[:500]!r}")

            system_spec = self.parse_and_validate(
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

        except Exception as exc:
            logger.error(f"[ClassDiagram] generate_complete_system FAILED: {exc}", exc_info=True)
            return self.generate_fallback_system()

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
        """Generate modifications for existing class diagram elements"""
        
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
                    mods = modification_spec['modifications']
                    actions_summary = ", ".join(m.get('action', '?') for m in mods)
                    target_names = set()
                    for m in mods:
                        t = m.get('target', {})
                        n = t.get('className') or t.get('attributeName') or t.get('methodName') or 'element'
                        target_names.add(n)
                    modification_spec['message'] = f"Applied {len(mods)} modifications ({actions_summary}) to {', '.join(target_names)}"
                else:
                    mod_action = modification_spec['modification'].get('action', 'modification')
                    target = modification_spec['modification'].get('target', {})
                    target_name = target.get('className') or target.get('attributeName') or target.get('methodName') or 'element'
                    modification_spec['message'] = f"Applied {mod_action} to {target_name}"

            logger.info(
                f"[ClassDiagram] Modification spec: "
                f"batch={'modifications' in modification_spec}, "
                f"keys={list(modification_spec.keys())}"
            )
            
            return modification_spec
            
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
