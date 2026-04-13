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
from ..core.prompt_fragments import POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE
from schemas import SingleClassSpec, SystemClassSpec, ClassModificationResponse
from utilities.model_context import detailed_model_summary

logger = logging.getLogger(__name__)


class ClassDiagramHandler(BaseDiagramHandler):
    """Handler for Class Diagram generation"""

    def get_diagram_type(self) -> str:
        return "ClassDiagram"

    def get_system_prompt(self) -> str:
        return f"""You are a UML modeling expert. Create a focused class specification based on the user's request.

RULES:
1. Include everything the user asks for, then add relevant domain attributes to make the class thorough.
2. Create AS MANY attributes as needed based on what makes sense for the class.
3. Methods: Generally SKIP methods unless the user asks for them. Only include core domain methods (e.g., BankAccount.withdraw(), Order.calculateTotal()). Never include getters/setters.
4. If the user just says "create X class", generate relevant attributes and typically NO methods.
5. Use proper naming: PascalCase for classes, camelCase for attributes/methods.
6. {POSITION_DISCLAIMER}

Examples of expected richness:
- "create User class" → id, username, email, password (4 attributes, 0-1 method)
- "create Product with inventory" → id, name, price, stockQuantity, supplier (5+ attributes)
- "create BankAccount with deposit method" → accountNumber, balance, owner + methods: deposit, withdraw"""

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a single class element with structured outputs and deterministic positioning."""

        system_prompt = self.get_system_prompt()
        user_prompt = f"Create a class specification for: {user_request}"

        logger.info(f"[ClassDiagram] generate_single_element called with: {user_request!r}")

        try:
            parsed = self.predict_structured(
                user_prompt,
                SingleClassSpec,
                system_prompt=system_prompt,
            )
            simple_spec = parsed.model_dump()

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
            logger.error(f"❌ [ClassDiagram] generate_single_element LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't generate that class. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"❌ [ClassDiagram] generate_single_element FAILED: {exc}", exc_info=True)
            return self._error_response(
                "I had trouble generating that class. Could you try rephrasing?",
                code="generation_error",
            )

    def _get_system_generation_prompt(self) -> str:
        """Return the system prompt for complete class diagram generation."""
        return f"""You are a UML modeling expert. Create a COMPLETE, well-structured class diagram system.

Before generating, think through:
- What are the core domain entities (classes) needed?
- What attributes does each class need? Be thorough — include IDs, timestamps, status fields.
- What relationships connect them? What type and what multiplicities?
- Is there an inheritance hierarchy that makes sense?
- Are relationships complete? They are the most commonly missed element.

RULES:
1. Include all the classes, relationships, and concepts the user asks for. Then flesh out each class with thorough attributes (IDs, timestamps, status fields where appropriate).
2. Create AS MANY classes as needed for a complete system.
3. Each class should have 3-5+ attributes. Don't create stub classes.
4. When creating Enumerations (isEnumeration=true), list enum values as attributes (name only, no type needed). When another class has an attribute whose type is that enumeration, set the attribute's type to the enum's PascalCase name (e.g., type="OrderStatus", NOT "str" or "String").
5. Methods: Generally SKIP methods unless the user asks. Only include 1-2 core domain methods per class MAX. Never include getters/setters.
6. Relationships are CRITICAL — always include meaningful connections. Use Association (general), Inheritance (is-a, sparingly), Composition (strong has-a), Aggregation (weak has-a), Realization (interface).
7. ALWAYS include multiplicities on relationships (1, 0..1, 0..*, 1..*).
8. Generate both associations AND inheritance where appropriate (e.g., SavingsAccount/CheckingAccount → inherit from Account).
9. Use proper naming: PascalCase for classes, camelCase for attributes/methods.
10. {POSITION_DISCLAIMER}
11. Methods default to implementationType "none" (UML signature only, no code). ONLY generate code in the 'code' field when the user explicitly asks for it. Supported types: 'code' for Python (e.g., "implement in Python", "add Python code"), 'bal' for BESSER Action Language (e.g., "implement in BAL", "use action language"). BAL syntax: def method_name(param: type) -> return_type {{ statements; }}. Python syntax: standard def with self parameter.

Examples:
- E-commerce: User, Product, Order, Payment, ShoppingCart with associations and multiplicities
- Library: Book, Author, Member, Loan with inheritance (DigitalBook extends Book) and compositions
- Banking: Account, Customer, Transaction, Branch with aggregations and multiplicities"""

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a complete class diagram with two-pass structured outputs, domain patterns,
        validation-feedback loop, and deterministic layout."""

        system_prompt = self._get_system_generation_prompt()

        logger.info(f"[ClassDiagram] generate_complete_system called with: {user_request!r}")

        try:
            # --- Two-pass structured: reason first, then produce validated Pydantic model ---
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

            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                response_schema=SystemClassSpec,
            )
            system_spec = parsed.model_dump()

            logger.info(
                f"[ClassDiagram] Structured system spec: "
                f"{len(system_spec.get('classes', []))} classes, "
                f"{len(system_spec.get('relationships', []))} relationships"
            )

            # TODO: Disabled for now — the extra LLM round-trip adds 2-4s latency
            # and the structured output schema already enforces correctness.
            # Re-enable once we have a faster validation strategy (e.g. rule-based
            # checks instead of an LLM call).
            # system_spec = self.validate_and_refine(
            #     system_spec,
            #     user_request=user_request,
            #     diagram_type="ClassDiagram",
            # )

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
            logger.error(f"❌ [ClassDiagram] generate_complete_system LLM FAILED: {exc}")
            return self._incremental_system_fallback(user_request, existing_model)
        except Exception as exc:
            logger.error(f"❌ [ClassDiagram] generate_complete_system FAILED: {exc}", exc_info=True)
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

        system_prompt = (
            """You are a UML modeling expert. Modify an existing class diagram.

COMMON ACTIONS:
- add_class — create a NEW class with attributes and methods. Put className, attributes, and methods in "changes".
- modify_class — rename a class or change its properties
- add_attribute / modify_attribute — add or change an attribute on a class
- add_method / modify_method — add or change a method on a class
- add_relationship — create a NEW connection between two classes
- modify_relationship — change an EXISTING relationship (multiplicity, type, name)
- remove_element — delete a class, attribute, method, or relationship

ADVANCED ACTIONS (for structural refactoring):
- extract_class, split_class, merge_classes, promote_attribute, add_enum

CRITICAL — READ CAREFULLY:
- The CURRENT MODEL is provided below. NEVER re-create anything that already exists.
- The conversation history is also provided. If it says you JUST created something, it EXISTS. Do NOT re-create it.
- ONLY output modifications for what the user asks RIGHT NOW. Never repeat past operations.
- If the user's message is short/ambiguous (e.g., "ok and X?", "also Y"), interpret it as ADDING to the most recently discussed element.

NAMING: Class names MUST be exactly ONE word in PascalCase: "User", "Book", "Order", "Payment".
NEVER concatenate words like "UserLibraryUser", "BookReading", "OrderPayment". Just "User", "Reading", "Payment".

KEY RULES:
1. Use exact names from the current model in "target".
2. Put what should change in "changes". Only include fields that differ.
3. """
            + REMOVE_ELEMENT_RULE
            + """
4. Multiple changes → use "modifications" array.
5. RENAME: single modify_class only. Relationships update automatically.
6. DELETE a class: you MUST include remove_element entries for the class AND for EVERY relationship connected to it. If you only mention the removal in your message but don't include the remove_element actions, the class WILL NOT be removed. Example: to delete "Address" with 2 relationships → 3 remove_element entries (1 for the class + 2 for relationships).
7. modify_relationship = update existing. add_relationship = brand new.
8. add_class: set target.className and put className, attributes[], methods[] in "changes".
9. When adding new classes, add relationships connecting them to existing classes. Include multiplicities. This is critical — isolated classes with no relationships are useless.

ENUMERATION RULES:
- Create enum: add_class with isEnumeration=true. Enum values are attributes with name only (NO type field).
- Add value to EXISTING enum: add_attribute with target.className set to THE ENUM NAME (not another class).
  Example: if "Priority" enum exists and user says "add Critical" → add_attribute with target.className="Priority", changes.name="Critical" (NO type).
- Use enum as attribute type: add_attribute with changes.type set to the enum's PascalCase name.
  Example: add_attribute with target.className="Task", changes.name="priority", changes.type="Priority".

Examples:
- "rename User to Customer" → ONE modify_class (no relationship changes needed)
- "add email to User" → add_attribute target.className="User", changes.name="email", changes.type="String"
- "add name, age, email to Person" → modifications array with 3 add_attribute entries
- "connect Order to Customer" → add_relationship (Association)
- "change multiplicity to many" → modify_relationship
- "delete the Address class" → modifications array: [remove_element with target.className="Address", remove_element with target.relationshipName="..." for EACH relationship involving Address]. You MUST include ALL of these or the class stays on the diagram.
- "add a User class with name and email" → add_class with target.className="User", changes.className="User", changes.attributes=[{name:"name",type:"String"},{name:"email",type:"String"}]
- "create an OrderStatus enum with PENDING, SHIPPED, DELIVERED" → add_class with isEnumeration=true, changes.attributes=[{name:"PENDING"},{name:"SHIPPED"},{name:"DELIVERED"}]
- "add status attribute of type OrderStatus to Order" → add_attribute with target.className="Order", changes.name="status", changes.type="OrderStatus"
- "create a Priority enum with Low, Medium, High" → add_class isEnumeration=true, className="Priority", attributes=[{name:"Low"},{name:"Medium"},{name:"High"}]
- "add Critical to the Priority enum" → add_attribute target.className="Priority", changes.name="Critical" (NO type)
- "add priority attribute to Task" → add_attribute target.className="Task", changes.name="priority", changes.type="Priority"
- "I also want to store users and books" → multiple add_class entries + add_relationship entries to connect them to existing classes"""
        )

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
            def _strip_spurious_relationship_mods(mod_list):
                """Strip modify_relationship entries that accompany a modify_class
                rename -- relationships are linked by ID and update automatically."""
                has_class_rename = any(
                    m.get("action") == "modify_class" and m.get("changes", {}).get("name")
                    for m in mod_list
                )
                if has_class_rename:
                    before = len(mod_list)
                    mod_list = [m for m in mod_list if m.get("action") != "modify_relationship"]
                    if len(mod_list) < before:
                        logger.info(
                            f"[ClassDiagram] Stripped {before - len(mod_list)} "
                            "spurious modify_relationship entries from class rename"
                        )

                # Normalize remove_element targets — some LLMs misplace the class
                # name into other fields or leave className null. Promote any
                # non-null string in target/changes to className when we're
                # clearly removing a class (no relationship/attribute/method hint).
                for mod in mod_list:
                    if mod.get("action") != "remove_element":
                        continue
                    target = mod.get("target") or {}
                    if not isinstance(target, dict):
                        continue
                    # Skip if already has a specific identifier
                    if any(target.get(k) for k in ("className", "classId",
                                                    "relationshipId", "relationshipName",
                                                    "attributeId", "attributeName",
                                                    "methodId", "methodName")):
                        continue
                    # Try to find a class name in any target or changes string field
                    changes = mod.get("changes") or {}
                    for source in (target, changes if isinstance(changes, dict) else {}):
                        for value in source.values():
                            if isinstance(value, str) and value.strip():
                                target["className"] = value.strip()
                                logger.info(
                                    f"[ClassDiagram] Normalized remove_element target: "
                                    f"promoted '{value}' to className"
                                )
                                break
                        if target.get("className"):
                            break
                    mod["target"] = target

                # Deduplicate remove_element entries that target the same class.
                # Some LLMs emit multiple removals for the same class name (once
                # for the class itself, once for each relationship it participates
                # in, all with className only). The frontend applies them in
                # sequence — the second one can't find the already-removed class
                # and used to throw. Keep only the first occurrence.
                seen_class_removes: set[str] = set()
                deduped: list = []
                for mod in mod_list:
                    if mod.get("action") == "remove_element":
                        target = mod.get("target") or {}
                        cn = (target.get("className") or "").strip().lower()
                        # Only dedupe class-level removals (no attribute/method/relationship)
                        is_class_only = (
                            cn
                            and not target.get("attributeName") and not target.get("attributeId")
                            and not target.get("methodName") and not target.get("methodId")
                            and not target.get("relationshipName") and not target.get("relationshipId")
                        )
                        if is_class_only:
                            if cn in seen_class_removes:
                                logger.info(
                                    f"[ClassDiagram] Dropped duplicate remove_element for class '{cn}'"
                                )
                                continue
                            seen_class_removes.add(cn)
                    deduped.append(mod)
                return deduped

            def _expand_refactoring(handler, spec):
                """Expand refactoring actions into primitive modifications."""
                if handler._is_refactoring_action(spec):
                    logger.info("[ClassDiagram] Detected refactoring action, expanding into primitives")
                    spec = handler._expand_refactoring_actions(spec, current_model)
                return spec

            modification_spec = self._execute_modification(
                full_prompt, "", ClassModificationResponse,
                post_processor=_strip_spurious_relationship_mods,
                spec_processor=_expand_refactoring,
            )

            logger.info(
                f"[ClassDiagram] Modification spec: "
                f"batch={'modifications' in modification_spec}, "
                f"keys={list(modification_spec.keys())}"
            )

            return modification_spec

        except LLMPredictionError as exc:
            logger.error(f"❌ [ClassDiagram] generate_modification LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't process that modification. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"❌ [ClassDiagram] generate_modification FAILED: {exc}", exc_info=True)
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
    # Refactoring Action Expansion
    # ------------------------------------------------------------------

    _REFACTORING_ACTIONS = frozenset({
        "extract_class", "split_class", "merge_classes",
        "promote_attribute", "add_enum",
    })

    def _is_refactoring_action(self, spec: Dict[str, Any]) -> bool:
        """Return True if the modification spec contains a refactoring action."""
        mod = spec.get("modification", {})
        if isinstance(mod, dict) and mod.get("action") in self._REFACTORING_ACTIONS:
            return True
        for m in spec.get("modifications", []):
            if isinstance(m, dict) and m.get("action") in self._REFACTORING_ACTIONS:
                return True
        return False

    def _expand_refactoring_actions(
        self, spec: Dict[str, Any], current_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Expand high-level refactoring actions into batches of primitive modifications.

        Each refactoring action (extract_class, split_class, etc.) is decomposed
        into a list of standard modification primitives (add_attribute,
        remove_element, add_relationship, etc.) that the frontend already knows
        how to apply.

        Returns a new spec with all refactoring actions expanded.
        """
        # Collect all inner modifications (single or batch)
        if "modifications" in spec and isinstance(spec["modifications"], list):
            raw_mods = list(spec["modifications"])
        elif "modification" in spec and isinstance(spec["modification"], dict):
            raw_mods = [spec["modification"]]
        else:
            return spec

        expanded: List[Dict[str, Any]] = []
        messages: List[str] = []

        for mod in raw_mods:
            action = mod.get("action", "")
            if action == "extract_class":
                sub_mods, msg = self._expand_extract_class(mod, current_model)
                expanded.extend(sub_mods)
                messages.append(msg)
            elif action == "split_class":
                sub_mods, msg = self._expand_split_class(mod, current_model)
                expanded.extend(sub_mods)
                messages.append(msg)
            elif action == "merge_classes":
                sub_mods, msg = self._expand_merge_classes(mod, current_model)
                expanded.extend(sub_mods)
                messages.append(msg)
            elif action == "promote_attribute":
                sub_mods, msg = self._expand_promote_attribute(mod, current_model)
                expanded.extend(sub_mods)
                messages.append(msg)
            elif action == "add_enum":
                sub_mods, msg = self._expand_add_enum(mod, current_model)
                expanded.extend(sub_mods)
                messages.append(msg)
            else:
                # Not a refactoring action; pass through as-is
                expanded.append(mod)

        result = dict(spec)
        # Always use batch format for expanded results
        result.pop("modification", None)
        result["modifications"] = expanded
        if messages:
            result["message"] = " ".join(messages)
        return result

    def _expand_extract_class(
        self, mod: Dict[str, Any], current_model: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Expand an extract_class action into primitive modifications.

        Produces:
        1. An inject_element for the new class (with the extracted attributes)
        2. A remove_element for each extracted attribute on the source class
        3. An add_relationship between source and new class
        """
        source_class = mod.get("sourceClass", "")
        new_class = mod.get("newClass", "NewClass")
        attributes = mod.get("attributes", [])
        rel_type = mod.get("relationshipType", "Composition")

        primitives: List[Dict[str, Any]] = []

        # Resolve attribute details from current model if available
        extracted_attrs = self._resolve_attributes(source_class, attributes, current_model)

        # 1. Create the new class with extracted attributes
        primitives.append({
            "action": "add_class",
            "target": {"className": new_class},
            "changes": {
                "className": new_class,
                "attributes": extracted_attrs,
                "methods": [],
            },
        })

        # 2. Remove each extracted attribute from the source class
        for attr_name in attributes:
            primitives.append({
                "action": "remove_element",
                "target": {
                    "className": source_class,
                    "attributeName": attr_name,
                },
            })

        # 3. Add a relationship from source to new class
        primitives.append({
            "action": "add_relationship",
            "target": {
                "sourceClass": source_class,
                "targetClass": new_class,
            },
            "changes": {
                "relationshipType": rel_type,
                "sourceMultiplicity": "1",
                "targetMultiplicity": "1",
                "name": f"has{new_class}",
            },
        })

        msg = (
            f"Extracted **{new_class}** from **{source_class}** with "
            f"attributes: {', '.join(f'`{a}`' for a in attributes)}."
        )
        return primitives, msg

    def _expand_split_class(
        self, mod: Dict[str, Any], current_model: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Expand a split_class action into primitive modifications.

        Produces:
        1. An add_class for each new class (with its subset of attributes)
        2. Optionally, add_relationship (Inheritance) from each new class to the source
        """
        source_class = mod.get("sourceClass", "")
        new_classes = mod.get("newClasses", [])
        inherit_from = mod.get("inheritFrom", "")

        primitives: List[Dict[str, Any]] = []
        class_names: List[str] = []

        for cls_spec in new_classes:
            cls_name = cls_spec.get("className", "NewClass")
            class_names.append(cls_name)
            attr_names = cls_spec.get("attributes", [])

            # Resolve full attribute definitions from the current model
            resolved_attrs = self._resolve_attributes(source_class, attr_names, current_model)

            primitives.append({
                "action": "add_class",
                "target": {"className": cls_name},
                "changes": {
                    "className": cls_name,
                    "attributes": resolved_attrs,
                    "methods": [],
                },
            })

            # If inheritance is requested, each new class inherits from source
            if inherit_from:
                primitives.append({
                    "action": "add_relationship",
                    "target": {
                        "sourceClass": cls_name,
                        "targetClass": inherit_from,
                    },
                    "changes": {
                        "relationshipType": "Inheritance",
                        "name": f"{cls_name}_extends_{inherit_from}",
                    },
                })

        msg = (
            f"Split **{source_class}** into {', '.join(f'**{n}**' for n in class_names)}"
            + (f" (inheriting from **{inherit_from}**)." if inherit_from else ".")
        )
        return primitives, msg

    def _expand_merge_classes(
        self, mod: Dict[str, Any], current_model: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Expand a merge_classes action into primitive modifications.

        Produces:
        1. An add_class for the merged target (union of all attributes)
        2. A remove_element for each source class being merged
        """
        classes_to_merge = mod.get("classes", [])
        target_name = mod.get("targetName", classes_to_merge[0] if classes_to_merge else "MergedClass")

        primitives: List[Dict[str, Any]] = []

        # Collect all attributes from all classes being merged
        merged_attrs: List[Dict[str, Any]] = []
        seen_attr_names: set = set()

        for cls_name in classes_to_merge:
            cls_attrs = self._get_class_attributes(cls_name, current_model)
            for attr in cls_attrs:
                attr_name = attr.get("name", "")
                if attr_name and attr_name not in seen_attr_names:
                    seen_attr_names.add(attr_name)
                    merged_attrs.append(attr)

        # 1. Create the merged class
        primitives.append({
            "action": "add_class",
            "target": {"className": target_name},
            "changes": {
                "className": target_name,
                "attributes": merged_attrs,
                "methods": [],
            },
        })

        # 2. Remove the original classes (except the target if it already exists)
        for cls_name in classes_to_merge:
            if cls_name != target_name:
                primitives.append({
                    "action": "remove_element",
                    "target": {"className": cls_name},
                })

        msg = (
            f"Merged {', '.join(f'**{c}**' for c in classes_to_merge)} "
            f"into **{target_name}** with {len(merged_attrs)} attribute(s)."
        )
        return primitives, msg

    def _expand_promote_attribute(
        self, mod: Dict[str, Any], current_model: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Expand a promote_attribute action into primitive modifications.

        Produces:
        1. An add_class for the new class (with the provided new attributes)
        2. A remove_element to remove the original attribute from the source class
        3. An add_relationship from source to the new class
        """
        source_class = mod.get("sourceClass", "")
        attribute = mod.get("attribute", "")
        new_class = mod.get("newClass", attribute.capitalize() if attribute else "PromotedClass")
        new_attributes = mod.get("newAttributes", [])

        primitives: List[Dict[str, Any]] = []

        # Ensure new attributes have required fields
        full_attrs: List[Dict[str, Any]] = []
        for attr in new_attributes:
            if isinstance(attr, dict):
                full_attrs.append({
                    "name": attr.get("name", "value"),
                    "type": attr.get("type", "String"),
                    "visibility": attr.get("visibility", "public"),
                })

        # If no attributes specified, create a sensible default
        if not full_attrs:
            full_attrs = [
                {"name": "id", "type": "String", "visibility": "public"},
                {"name": "value", "type": "String", "visibility": "public"},
            ]

        # 1. Create the new class
        primitives.append({
            "action": "add_class",
            "target": {"className": new_class},
            "changes": {
                "className": new_class,
                "attributes": full_attrs,
                "methods": [],
            },
        })

        # 2. Remove the original primitive attribute from the source class
        primitives.append({
            "action": "remove_element",
            "target": {
                "className": source_class,
                "attributeName": attribute,
            },
        })

        # 3. Add relationship from source class to the new class
        primitives.append({
            "action": "add_relationship",
            "target": {
                "sourceClass": source_class,
                "targetClass": new_class,
            },
            "changes": {
                "relationshipType": "Association",
                "sourceMultiplicity": "1",
                "targetMultiplicity": "1",
                "name": f"has{new_class}",
            },
        })

        msg = (
            f"Promoted `{attribute}` from **{source_class}** into a new "
            f"**{new_class}** class with {len(full_attrs)} attribute(s)."
        )
        return primitives, msg

    def _expand_add_enum(
        self, mod: Dict[str, Any], current_model: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Expand an add_enum action into primitive modifications.

        Produces:
        1. An add_class for the enumeration (with values as attributes)
        2. A modify_attribute for each class/attribute pair that should use the enum type
        """
        enum_name = mod.get("enumName", "NewEnum")
        values = mod.get("values", [])
        used_by = mod.get("usedBy", [])

        primitives: List[Dict[str, Any]] = []

        # 1. Create the enum as a class with <<enumeration>> stereotype
        enum_attrs = [
            {"name": v, "type": enum_name, "visibility": "public"}
            for v in values if isinstance(v, str)
        ]
        primitives.append({
            "action": "add_class",
            "target": {"className": enum_name},
            "changes": {
                "className": enum_name,
                "stereotype": "enumeration",
                "attributes": enum_attrs,
                "methods": [],
            },
        })

        # 2. Update each referencing attribute to use the enum type
        for usage in used_by:
            if not isinstance(usage, dict):
                continue
            cls_name = usage.get("className", "")
            attr_name = usage.get("attributeName", "")
            if cls_name and attr_name:
                primitives.append({
                    "action": "modify_attribute",
                    "target": {
                        "className": cls_name,
                        "attributeName": attr_name,
                    },
                    "changes": {
                        "type": enum_name,
                    },
                })

        msg = (
            f"Created enumeration **{enum_name}** with values: "
            f"{', '.join(f'`{v}`' for v in values)}"
        )
        if used_by:
            refs = ", ".join(
                f"**{u.get('className', '?')}**.`{u.get('attributeName', '?')}`"
                for u in used_by if isinstance(u, dict)
            )
            msg += f" (used by {refs})"
        msg += "."
        return primitives, msg

    # ------------------------------------------------------------------
    # Attribute Resolution Helpers
    # ------------------------------------------------------------------

    def _resolve_attributes(
        self,
        class_name: str,
        attr_names: List[str],
        current_model: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Resolve attribute names to full attribute dicts from the current model.

        If the attribute exists in *current_model*, its type and visibility are
        preserved.  Otherwise a sensible default (type String, visibility public)
        is used.
        """
        model_attrs = self._get_class_attributes(class_name, current_model)
        model_map: Dict[str, Dict[str, Any]] = {}
        for attr in model_attrs:
            name = attr.get("name", "")
            if name:
                model_map[name] = attr

        resolved: List[Dict[str, Any]] = []
        for name in attr_names:
            if name in model_map:
                resolved.append(dict(model_map[name]))
            else:
                resolved.append({
                    "name": name,
                    "type": "String",
                    "visibility": "public",
                })
        return resolved

    def _get_class_attributes(
        self, class_name: str, current_model: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return the list of attribute dicts for *class_name* from the current model.

        Searches through the model's elements dict to find the matching class.
        Returns an empty list if the class or model is not available.
        """
        if not isinstance(current_model, dict):
            return []
        elements = current_model.get("elements")
        if not isinstance(elements, dict):
            return []
        for el in elements.values():
            if not isinstance(el, dict):
                continue
            if el.get("type") == "Class" and el.get("name") == class_name:
                attrs = el.get("attributes", [])
                if isinstance(attrs, list):
                    return attrs
                # Sometimes attributes are stored as a dict keyed by ID
                if isinstance(attrs, dict):
                    return list(attrs.values())
        return []

    # ------------------------------------------------------------------
    # Impact Analysis Helpers
    # ------------------------------------------------------------------

    def _build_impact_context(self, model: Optional[Dict[str, Any]]) -> str:
        """Build a relationship dependency map for modification impact analysis.

        For each class, lists all relationships it participates in so the LLM
        knows which relationships to remove when deleting a class.
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
        lines = ["Relationship dependencies (only relevant when REMOVING a class — renames cascade automatically):"]
        for class_name, dep_list in deps.items():
            if dep_list:
                lines.append(f"  {class_name}: {', '.join(dep_list)}")

        return "\n".join(lines) if len(lines) > 1 else ""
