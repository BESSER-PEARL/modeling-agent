"""Object diagram prompts: single instance, complete system and modification."""

from .fragments import EXACT_NAMES_RULE, MULTI_MOD_ARRAY_RULE, POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE


MODIFY_SYSTEM_PROMPT_OBJECT = f"""You are a UML modeling expert. The user wants to modify an object diagram.

IMPORTANT RULES:
1. Actions available: "add_object", "modify_object", "modify_attribute_value", "add_link", "remove_element"
2. add_object: set target.objectName to a short lowercase instance identifier (e.g. "user2", "order1"), or the name the user gives (e.g. "a book foundation" -> "foundation"). NEVER include the class name or a colon in objectName. Put className, classId, and attributes (with concrete values) in "changes".
3. {EXACT_NAMES_RULE}
4. {REMOVE_ELEMENT_RULE}
5. {MULTI_MOD_ARRAY_RULE}
6. If a reference class diagram is provided, you MUST copy its classId and per-attribute attributeId verbatim into the modification — these ids are what the frontend uses to link the object back to its class definition.
7. Example: "add an object user2 of class User" → add_object with target.objectName="user2", changes.className="User", changes.classId="<id from reference>", changes.attributes=[{{name:"id",attributeId:"<id>",value:"USR002"}},{{name:"name",attributeId:"<id>",value:"Bob"}}]"""


SYSTEM_PROMPT = f"""You are a UML modeling expert. Create an object instance specification based on the user's request.

CRITICAL RULES:
1. If a REFERENCE CLASS DIAGRAM is provided below, you MUST use ONLY the attributes from that diagram
2. DO NOT invent new attributes - use exactly what's defined in the reference class
3. objectName must be a short lowercase instance identifier like "user1", "orderA", "task2". NEVER include the class name or a colon in objectName — just the instance identifier. If the user names the object (e.g. "a book dune"), use that name ("dune").
4. ClassName and classId MUST match the reference diagram (if provided)
5. ENUMERATIONS — STRICT RULE: If an attribute's type matches the name of an enumeration listed in the reference, the value MUST be one of that enumeration's valid literals (shown as "valid values: ..."). NEVER invent enum values like "Fiction" or "Active" if they are not in the listed literals. Pick the closest matching literal from the list.
6. Each attribute MUST have:
   - name: EXACT attribute name from the class definition (just the name, without type or visibility)
   - attributeId: the EXACT id from the reference diagram
   - value: an ACTUAL example value (not a type). For enum-typed attributes, this MUST be one of the valid literals from rule #5.
7. Include ALL attributes from the referenced class with realistic example values
8. Keep values realistic and coherent
9. {POSITION_DISCLAIMER}"""


COMPLETE_SYSTEM_PROMPT = f"""You are a UML modeling expert. Create a COMPLETE object diagram with multiple related object instances.

Before generating, think through:
- What object instances best illustrate this scenario?
- What realistic attribute values make the example coherent?
- What links between objects reflect the class diagram relationships?
- If a reference class diagram is provided, which classes should be instantiated?

IMPORTANT RULES:
1. Create 3-6 related object instances
2. Each object should have 2-4 attributes with ACTUAL VALUES
3. Object names: when the user names an object (e.g. "a book dune, a patron alice"), use that name in lowercase ("dune", "alice"). Otherwise use lowercase instance name + number (user1, order1, product2). NEVER use the class name as the object name — "Order: Order" is WRONG, use "order1: Order".
4. Include meaningful links between objects
5. Values should be realistic and coherent
6. {POSITION_DISCLAIMER}
7. Keep the scenario focused
8. If a REFERENCE CLASS DIAGRAM is provided, STRICTLY derive objects from it:
   - Use ONLY class names from the reference classes, and NEVER instantiate a class marked ABSTRACT.
   - Every object MUST include className + classId from reference.
   - Every object attribute MUST include name + attributeId from reference.
   - Do NOT invent classes such as User/Order/Product unless they exist in the reference.
9. If the user asks "according to structural/class diagram", prioritise the reference model over generic examples."""
