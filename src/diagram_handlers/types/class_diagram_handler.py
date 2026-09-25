"""
Class Diagram Handler
Handles generation of UML Class Diagrams
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional

from ..core.base_handler import (
    BaseDiagramHandler,
    LLMPredictionError,
    SINGLE_CLASS_REQUIRED,
    SINGLE_CLASS_OPTIONAL,
    SYSTEM_CLASS_REQUIRED,
    SYSTEM_CLASS_OPTIONAL,
)
from ..core.prompt_fragments import (
    CHANGES_FIELD_RULE,
    DELETE_CLASS_CASCADE_RULE,
    ENUM_RULES_BLOCK,
    EXACT_NAMES_RULE,
    MODIFY_CRITICAL_BLOCK,
    MULTI_MOD_ARRAY_RULE,
    NAMING_PASCAL_RULE,
    OCL_CONSTRAINT_BLOCK,
    OCL_EXAMPLES_BLOCK,
    POSITION_DISCLAIMER,
    REMOVE_ELEMENT_RULE,
    RENAME_CASCADES_RULE,
)
from model_config import MODEL_GENERATION_LARGE, MODEL_GENERATION_SMALL, MODEL_REASONING
from schemas import SingleClassSpec, SystemClassSpec, ClassModificationResponse
from schemas.class_diagram import RecoveredEnumerationsSpec
from schemas.compact_class_diagram import (
    COMPACT_ENCODING_RULES,
    CompactSystemClassSpec,
    expand_compact_spec,
)
from utilities.model_context import detailed_model_summary

logger = logging.getLogger(__name__)

# Compact structured output for complete-system generation: same model, same
# modeling rules, ~2.4x faster (see schemas/compact_class_diagram.py). Kill switch for rollback without a code
# change: BESSER_AGENT_COMPACT_SPEC=0.
COMPACT_SPEC_ENABLED = os.environ.get("BESSER_AGENT_COMPACT_SPEC", "1") != "0"


_CLASS_ACTIONS_BLOCK = """COMMON ACTIONS:
- add_class — create a NEW class with attributes and methods. Put className, attributes, and methods in "changes".
- modify_class — rename a class or change its properties
- add_attribute / modify_attribute — add or change an attribute on a class
- add_method / modify_method — add or change a method on a class
- add_relationship — create a NEW connection between two classes
- modify_relationship — change an EXISTING relationship (multiplicity, type, name)
- remove_element — delete a class, attribute, method, or relationship

ADVANCED ACTIONS (for structural refactoring):
- extract_class, split_class, merge_classes, promote_attribute, add_enum"""

_CLASS_KEY_RULES_BLOCK = f"""KEY RULES:
1. {EXACT_NAMES_RULE}
2. {CHANGES_FIELD_RULE}
3. {REMOVE_ELEMENT_RULE}
4. {MULTI_MOD_ARRAY_RULE}
5. {RENAME_CASCADES_RULE}
6. {DELETE_CLASS_CASCADE_RULE}
7. modify_relationship = update existing. add_relationship = brand new.
8. add_class: set target.className and put className, attributes[], methods[] in "changes".
9. When adding new classes, add relationships connecting them to existing classes. Include multiplicities. This is critical — isolated classes with no relationships are useless."""

_CLASS_EXAMPLES_BLOCK = """Examples:
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

# Built once at import time so the system message is byte-stable across calls
# — that's what lets OpenAI's prompt cache hit on every modification turn
# after the first.
MODIFY_SYSTEM_PROMPT_CLASS = "\n\n".join([
    "You are a UML modeling expert. Modify an existing class diagram.",
    _CLASS_ACTIONS_BLOCK,
    OCL_CONSTRAINT_BLOCK,
    MODIFY_CRITICAL_BLOCK,
    NAMING_PASCAL_RULE,
    _CLASS_KEY_RULES_BLOCK,
    ENUM_RULES_BLOCK,
    _CLASS_EXAMPLES_BLOCK,
    OCL_EXAMPLES_BLOCK,
])


# ---------------------------------------------------------------------------
# Duplicate association-end repair (deterministic — no LLM)
# ---------------------------------------------------------------------------
# The BESSER metamodel rejects a class with two association ends of the same
# name; the editor's validate-and-repair loop forwards that error verbatim in
# an "[auto-fix]" repair request. The LLM modification schema historically had
# no field for END/role names, so the model "fixed" it by renaming the
# relationship LABEL — a no-op for validation — and then claimed success.
# These constants + the ClassDiagramHandler._*duplicate_end* helpers repair the
# duplicate deterministically in the diagram-JSON shape the frontend applies
# (modify_relationship + changes.roleName → rel.target.role).

# Matches the metamodel's exact error copy (structural.py raises it; the
# validation endpoint and the editor forward it unchanged).
_DUP_END_ERROR_RE = re.compile(
    r"[Tt]he class '(?P<cls>[^']+)' cannot have two association ends "
    r"with the same name: '(?P<end>[^']+)'"
)

# Editor relationship types that create association ends (mirror of the
# backend converter's handling in class_diagram_processor.py).
_ASSOCIATION_END_REL_TYPES = frozenset({
    "ClassBidirectional", "ClassUnidirectional",
    "ClassComposition", "ClassAggregation",
})
_INHERITANCE_REL_TYPES = frozenset({"ClassInheritance"})

# Editor element types that can own association ends.
_CLASS_ELEMENT_TYPES = frozenset({"Class", "AbstractClass", "Interface"})

# A name that is safe as a BUML / generated-code identifier: it starts with a
# letter or underscore and contains only letters, digits, and underscores.
# Anything else — hyphens, spaces, dots, slashes — makes the metamodel reject
# the element during validation. For a CLASS name that rejection is
# unrecoverable for the editor's auto-fix loop: it emits a "rename X" repair,
# but X was never accepted as a class, so the modify handler can't find it and
# the loop spins. Sanitizing the name before it reaches the canvas prevents the
# loop.
_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
            # Single element → SMALL generation tier (latency-sensitive).
            parsed = self.predict_structured(
                user_prompt,
                SingleClassSpec,
                system_prompt=system_prompt,
                model=MODEL_GENERATION_SMALL,
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
- What attributes does each class need? When the user listed them, use THEIR list and stop there. When they did not, be thorough — include IDs, timestamps, and a status field only if that entity actually has states.
- What relationships connect them? What type and what multiplicities?
- Is there an inheritance hierarchy that makes sense?
- Are relationships complete? They are the most commonly missed element.

RULES:
1. Include all the classes, relationships, and concepts the user asks for. Where the user SPECIFIED a class's attributes, use exactly those plus an id — do not add status, timestamps or anything else they did not name. Where they only named the class, flesh it out with thorough domain attributes (IDs, timestamps, and a status field only if that entity has states).
2. SCOPE: match the diagram size to the request. A plain request ("create a library model") gets the CORE domain only: 6-10 classes. Do NOT add peripheral subsystems (notifications, audit logs, reporting, staffing, fines, reservations, branches...) unless the user names them. Only exceed 12 classes when the user explicitly asks for a "complete", "comprehensive", "detailed" or "enterprise" system, or lists that many entities themselves. A focused diagram the user can extend beats an overwhelming one.
3. A class the user described gets exactly what they described — never padded to hit a count. A class they only named should have 3-5+ attributes rather than being a stub.
4. When creating Enumerations (isEnumeration=true), list enum values as attributes (name only, no type needed). When another class has an attribute whose type is that enumeration, set the attribute's type to the enum's PascalCase name (e.g., type="OrderStatus", NOT "str" or "String"). An enumeration is used ONLY as an attribute type — NEVER create a relationship (Association, Composition, Aggregation, etc.) whose source or target is an enumeration. For example, for a Task with a status, add an attribute Task.status of type TaskStatus; do NOT add a relationship from Task to the TaskStatus enum.
5. METHODS — TWO DIFFERENT CASES, DO NOT CONFUSE THEM. (a) THE USER NAMES ACTIONS. Any phrasing that describes something an entity DOES ("a booking can be cancelled", "produces a bill", "offers five actions: ...", "the member borrows and returns books") is an explicit request. Emit EXACTLY those methods, on the class the user attached them to, using the user's own names and the user's own COUNT — an explicit list overrides the cap in (b) completely. Five stated actions means five methods, not two. (b) THE USER SAYS NOTHING ABOUT BEHAVIOUR. Then SKIP methods, or at most 1-2 obvious core domain methods per class. In BOTH cases: NEVER invent a method on a class the user said nothing about, and NEVER include getters/setters. Worked example: "A booking offers five actions: produce the bill, check the guest in, check the guest out, cancel the booking, and compute the amount due" => Booking gets produceBill(), checkIn(), checkOut(), cancel(), computeAmountDue() — all five — and Employee, Guest and Room get NO methods, because the user described none for them.
6. Connect the classes with meaningful relationships — isolated classes are the most common gap. Use Association (general), Inheritance (is-a, sparingly), Composition (strong has-a), Aggregation (weak has-a), Realization (interface).
7. MULTIPLICITIES COME FROM THE USER'S OWN WORDS — never default to 1-to-1. Always include multiplicities (1, 0..1, 0..*, 1..*), and when the user states a cardinality in prose, TRANSCRIBE IT rather than guessing. Read each side separately: "may appear on several X" / "may hold several" / "many" / "one or more" / "zero or multiple" => the MANY side is 0..* or 1..* (use 1..* only when the user says at least one); "exactly one" / "one ... who is responsible" / "belongs to one" => that side is 1; "may ... or none" / "at most one" / "never more than one" / "optional" => 0..1 or 0..*. Worked example: "a given guest may appear on several bookings" AND "each booking lists at least one guest" => Booking 0..* -- 1..* Guest. "an employee may be responsible for many bookings" AND "each booking is handled by exactly one employee" => Employee 1 -- 0..* Booking. "a booking may have a bill raised against it, but never more than one" => Booking 1 -- 0..1 Bill. CHECK YOURSELF: a 1--1 association is rare and means the two objects always exist together and are never shared. Before emitting 1--1, re-read the request; if either side is described with a plural ("several", "many", "multiple", "a list of"), it is NOT 1--1. Getting this wrong produces a UNIQUE foreign key in the generated database and makes the app reject the user's real data. WHICH END GETS WHICH NUMBER - THIS IS THE PART THAT GOES WRONG. A multiplicity written at an end says HOW MANY OF THAT END'S CLASS relate to ONE instance of the class at the other end. So put the 'many' number on the class you have many OF. Worked example, with the exact encoding: "each booking has one person acting as its contact" AND "any individual may hold several bookings at once, or none at all" => source=Booking with multiplicity "0..*", target=Person with multiplicity "1" - because for one Person there are 0..* Bookings, and for one Booking there is exactly 1 contact Person. Writing it the other way round (source=Booking "1", target=Person "0..*") says every Person has exactly one Booking, which is the opposite of what the user wrote. Observed live 2026-09-17: that inversion on the contact, guests and handledBy associations made PersonCreate require a booking and BookingCreate require a contact person, so neither could ever be created and the whole app was unusable. Before emitting each relationship, say to yourself: 'for ONE <target class>, how many <source class> are there?' - that number is the SOURCE multiplicity.
8. Generate both associations AND inheritance where appropriate (e.g., SavingsAccount/CheckingAccount → inherit from Account).
9. Use proper naming: PascalCase for classes, camelCase for attributes/methods.
10. {POSITION_DISCLAIMER}
11. Methods default to implementationType "none" (UML signature only, no code). ONLY generate code in the 'code' field when the user explicitly asks for it. Supported types: 'code' for Python (e.g., "implement in Python", "add Python code"), 'bal' for BESSER Action Language (e.g., "implement in BAL", "use action language"). BAL syntax: def method_name(param: type) -> return_type {{ statements; }}. Python syntax: standard def with self parameter.
12. ENUMERATIONS ARE NEVER RELATED. A relationship (in the relationships list) must connect two real classes — NEVER an enumeration. If a class has an enum-valued property (status, priority, type, role, category, etc.), model it as an ATTRIBUTE on that class whose type is the enum's PascalCase name — do NOT emit a relationship pointing at the enum. There must be ZERO relationships whose source or target is an isEnumeration=true class.
14. ATTRIBUTE TYPES ARE PRIMITIVES OR ENUMS — NEVER ANOTHER CLASS. An attribute's type must be a primitive (String, int, bool, float, Date, datetime, time) or an enumeration name. If a class "has a" another class you also model (a PointOfInterest has a Location, an Order has a Customer, a Trip has a start Location and an end Location), express it as a RELATIONSHIP (Association) between the two classes — give the relationship a role name (e.g. startLocation, endLocation) to distinguish multiple links to the same class. NEVER put the class's name in an attribute's "type". Value objects you define (Location, Address, Money, Coordinates, TimeRange) are classes: connect them with relationships, don't use them as attribute types.
16. STATUS VOCABULARIES ARE THE USER'S WORDS, AND N DIMENSIONS MEANS N ENUMERATIONS. When the user lists the values a status can take, the enumeration's literals are EXACTLY those values — never add a member the user did not name, never rename one, never "improve" the set. And when the user describes TWO OR MORE INDEPENDENT things that vary separately, emit ONE ENUMERATION PER DIMENSION, each with its own attribute on the class — never merge them into a single status. Worked example: "a booking has a commercial status: awaiting payment, confirmed or cancelled. Separately it has a physical status: not arrived, checked in or checked out" => TWO enumerations, BookingCommercialStatus (AWAITING_PAYMENT, CONFIRMED, CANCELLED) and BookingPhysicalStatus (NOT_ARRIVED, CHECKED_IN, CHECKED_OUT), and Booking gets BOTH attributes. Merging them into one BookingStatus is wrong twice over: it makes states that legitimately co-occur (a CONFIRMED booking whose guest has NOT_ARRIVED) unrepresentable, and it forces you to invent members like NO_SHOW or BOOKED that the user never mentioned. Tell-tale wording for a second dimension: "separately", "independently", "as well as", "at the same time", or simply two different sentences each introducing its own list of values.
17. VALUES THAT BELONG TO A PAIRING USE A NATIVE ASSOCIATION CLASS. A value that depends on BOTH endpoints belongs to the link, not to either endpoint. Declare a concrete class for the link itself in classes with ALL those per-link attributes, then put its name in associationClass on ONE direct Association between the two endpoint classes (compact encoding: ac). Example: Student--Course with associationClass="Enrollment", and Enrollment.grade — the grade is neither the student's nor the course's, it exists only for that pairing. Putting a per-link value on endpoint A overwrites it for every other B it is paired with; putting it on B cannot tell two different A's apart. Tell-tale wording: "for that <A> in that <B>", "may differ from the standard", "per <A> per <B>", "the agreed/actual <value>, which may differ from the standard one". Keep the user's cardinalities on that direct association. Do NOT replace it with two ordinary links through the association class, and do not add those redundant links. Use an ordinary entity with separate associations instead only when the spec needs independent identity/lifecycle or multiple distinct records for the same pair. When the spec names SEVERAL per-link values, put them ALL on that one link class; never keep some and move the rest onto either endpoint.
15. PARALLEL ASSOCIATIONS NEED DISTINCT NAMES. When the same two classes are connected by MORE THAN ONE relationship (e.g. a Doctor "works in" a Department AND "heads" a Department), give EACH of those relationships a distinct, meaningful name ("worksIn", "heads") — never leave two relationships between the same pair of classes unnamed or identically named. The relationship name becomes the association end's role name; missing or duplicate names collide into the same role and fail validation. NAME BOTH ENDS OF EVERY RELATIONSHIP, ALWAYS: "name" is the TARGET end's role and "sourceRole" is the SOURCE end's (compact encoding: l and ls). An end you leave blank is not neutral - it defaults to the lowercased class name, and those defaults collide silently ACROSS AN INHERITANCE HIERARCHY. Observed live 2026-09-17, and it aborted the whole generation before a single file was written: Booking(unnamed)->Person[contact] gave Person an end called 'booking', Booking(unnamed)->Employee[handledBy] gave Employee an end called 'booking', Employee inherits Person, and the run died with "The class 'Employee' cannot have two association ends with the same name: 'booking'". Guest had the identical clash. So: whenever a PARENT class and its SUBCLASSES are all associated with the same other class - a Person/Employee/Guest hierarchy that all relate to Booking is the textbook case - every one of those ends needs its own distinct name. Correct: Booking->Person with sourceRole='bookingsAsContact' and name='contact'; Booking->Employee with sourceRole='bookingsHandled' and name='handledBy'; Booking->Guest with sourceRole='bookingsStayedOn' and name='guests'. Check every class, INCLUDING the names it inherits, before you finish.
13. CONSTRAINTS (OCL) — ONLY FOR RULES NO FIELD FLAG OR MULTIPLICITY CAN EXPRESS. SINGLE-ATTRIBUTE UNIQUENESS IS NOT ONE OF THEM: when the user says one attribute identifies the object or must be unique — "every room is identified by its room number", "emails must be unique", "each book is known by its ISBN" — set isExternalId=true on THAT attribute (compact encoding: the '!' suffix, 'roomNumber: str!') and write NO constraint for it. That flag is what makes the generated database reject a second room "101"; an OCL invariant for it is ignored by the generators and the duplicate gets in. If the user EXPLICITLY states a business rule that multiplicities, attribute types and that flag cannot express — a limit beyond cardinality ("a speaker presents at most one session per time slot"), a value range ("age must be at least 18") or a value shape ("must be a valid email") — capture it in the "constraints" list as an OCL invariant in B-OCL syntax: context <ClassName> inv <name>: <expression>. The context MUST be one of the classes you created. Examples: "context Account inv positiveBalance: self.balance >= 0"; "context Speaker inv oneSessionPerSlot: self.sessions->forAll(s1, s2 | s1 <> s2 implies s1.timeSlot <> s2.timeSlot)". CRITICAL: capture ONLY rules the user actually stated. If the user stated no such rule, leave "constraints" EMPTY — NEVER invent constraints. BUT "EXPLICITLY STATED" MEANS STATED IN ORDINARY PROSE — a rule does NOT have to be labelled "constraint", "invariant" or "rule", and does NOT have to be written in OCL. Every one of these phrasings is an explicit statement you must capture: "must not exceed", "cannot be more than", "may not overlap", "cannot be double-booked", "must be a valid <X>", "at least", "at most", "must not be before/after", "only if". Worked examples from a hotel request: "the total number of guests must not exceed the combined capacity of the rooms booked" => context Booking inv guestsWithinCapacity: self.guestCount <= self.rooms->collect(capacity)->sum(); "a room cannot be double-booked for overlapping dates" => context Room inv noOverlappingBookings: self.bookings->forAll(b1, b2 | b1 <> b2 implies b1.departureDate <= b2.arrivalDate or b2.departureDate <= b1.arrivalDate); "the arrival date must not be after the departure date" => context Booking inv arrivalBeforeDeparture: self.arrivalDate <= self.departureDate; "email and phone must be valid" => context Guest inv validEmail: self.email.matches('^[^\\s@]+@[^\\s@]+\\.[A-Za-z]{{2,}}$') and context Guest inv validPhone: self.phone.matches('^\\+?[0-9]{{7,15}}$'). A SHAPE CONSTRAINT IS ANCHORED WITH ^ AND $ AND NEVER ACCEPTS WHITESPACE unless the user said so — an unanchored pattern let "spaces in@email.com" through in a live run. Before you finish, re-read the request for these phrasings — four stated rules producing an empty constraints list is a failure, not caution.

18. A VALUE THE USER SAYS IS NOT ENTERED IS DERIVED. When the request says a value is "followed automatically", "never set by hand", "neither is set by hand", "not typed in", "worked out from", "computed from" or "calculated", mark that attribute DERIVED (compact encoding: the '/' prefix, '/totalPrice: float'). That flag is what keeps the value out of the generated create form and makes the backend compute it; an unmarked one ships as an ordinary stored field that the user is asked to type in, which is exactly what the request said must not happen. Worked examples: "The status of a task is followed automatically, and it is never set by hand" => /status: TaskStatus. "The total amount of the order is likewise not typed in. It is worked out from the quantities and agreed unit prices" => /totalAmount: float. "Two states are followed for every booking, and neither is set by hand" => BOTH state attributes are derived. Mark ONLY values the request describes this way - never guess that an ordinary field is computed.

Examples:
- E-commerce: User, Product, Order, Payment, ShoppingCart with associations and multiplicities
- Library: Book, Author, Member, Loan with inheritance (DigitalBook extends Book) and compositions
- Banking: Account, Customer, Transaction, Branch with aggregations and multiplicities"""

    def generate_complete_system(
        self,
        user_request: str,
        existing_model: Dict[str, Any] = None,
        raw_request: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a complete class diagram with two-pass structured outputs, domain patterns,
        validation-feedback loop, and deterministic layout.

        ``raw_request`` is the original user message before context enrichment
        (conversation history / workspace block); it drives the fast-path
        length check and keeps the reasoning prompt lean. Falls back to the
        full ``user_request`` when not provided.
        """

        system_prompt = self._get_system_generation_prompt()

        logger.info(f"[ClassDiagram] generate_complete_system called with: {user_request!r}")

        try:
            # --- Two-pass structured: reason first, then produce validated Pydantic model ---
            # The reasoning prompt uses the raw request only; the enriched
            # context (history + workspace block) reaches the structured pass
            # via ``user_request`` exactly once.
            reasoning_prompt = (
                "You are a UML domain modeling expert. Think step by step about "
                "the following system request and plan the class diagram design.\n\n"
                f"User Request: {raw_request or user_request}\n\n"
                "Analyze:\n"
                "1. What are the core domain entities (classes) needed? Match the "
                "scope to the request: a plain request gets the CORE domain only "
                "(6-10 classes); only plan a bigger model when the user explicitly "
                "asks for a comprehensive/complete system or names many entities.\n"
                "2. What attributes does each class need? (be thorough)\n"
                "3. What relationships connect these classes? What type (Association, "
                "Composition, Aggregation, Inheritance)? For each one state the two "
                "directions as SEPARATE sentences in this exact form — 'for one X "
                "there are <n> Y' and 'for one Y there are <n> X' — quoting the "
                "user's words for each. Do NOT use dash shorthand like 'X 1--* Y'; "
                "writing one number and mirroring the other is how the ends get "
                "swapped. If the same two classes need MORE THAN ONE relationship, "
                "plan a distinct, meaningful name for each (e.g. 'worksIn' vs "
                "'heads') — and if one fact is simply stated twice in the request "
                "(from each side), that is ONE relationship, not two.\n"
                "4. Are there any association classes needed (e.g., Enrollment between "
                "Student and Course with grade)?\n"
                "5. Is there any inheritance hierarchy that makes sense?\n"
                "6. Did the user say one attribute identifies an object or must be "
                "unique ('identified by its room number')? Plan isExternalId on "
                "that attribute, not an OCL rule. Did they EXPLICITLY state any "
                "other business rule that needs an OCL constraint (a limit beyond "
                "cardinality, a value range, a value shape)? Only note rules the "
                "user actually stated — do NOT invent any.\n"
                "7. Does the request say any value is NOT entered by the "
                "user — 'followed automatically', 'never set by hand', "
                "'not typed in', 'worked out from', 'computed from'? "
                "Name each one and say what it is computed from: those "
                "attributes are DERIVED. And does it restrict any value to "
                "a CLOSED SET of named options? Each such set is its own "
                "enumeration, declared as a class, with exactly the user's "
                "values in the order given.\n\n"
                "Provide a clear design analysis. Be thorough about relationships — "
                "they are the most commonly missed element. Do NOT pad the design "
                "with peripheral subsystems the user didn't ask for."
            )

            # Complete-system generation is where diagram quality is the
            # product → LARGE tier; reasoning pass on the REASONING tier.
            if COMPACT_SPEC_ENABLED:
                parsed = self.predict_two_pass_structured(
                    user_request=user_request,
                    system_prompt=system_prompt + COMPACT_ENCODING_RULES,
                    reasoning_prompt=reasoning_prompt,
                    response_schema=CompactSystemClassSpec,
                    raw_request=raw_request,
                    model=MODEL_GENERATION_LARGE,
                    reasoning_model=MODEL_REASONING,
                )
                # Deterministic expansion back to the canonical spec — guards,
                # layout, and the frontend payload below are untouched.
                parsed = expand_compact_spec(parsed)
            else:
                parsed = self.predict_two_pass_structured(
                    user_request=user_request,
                    system_prompt=system_prompt,
                    reasoning_prompt=reasoning_prompt,
                    response_schema=SystemClassSpec,
                    raw_request=raw_request,
                    model=MODEL_GENERATION_LARGE,
                    reasoning_model=MODEL_REASONING,
                )
            system_spec = parsed.model_dump()

            # Guard (runs first): rewrite any class/enum/member name that isn't a
            # valid identifier (hyphens, spaces, …). An invalid CLASS name fails
            # metamodel validation and sends the editor's auto-fix loop into an
            # unrecoverable spin (it can't rename a class that was never accepted).
            # Fixing names here — and every reference to them — keeps the rest of
            # the pipeline and the injected model clean.
            self._sanitize_identifier_names(system_spec)

            # Guard: never ship a relationship whose endpoint is an enumeration.
            # Rewrite any such relationship into an enum-typed attribute on the
            # non-enum side (or drop it). Enums are attribute types, not related.
            self._rewrite_enum_relationships(system_spec)

            # Guard: an attribute may reference an enumeration the model
            # forgot to declare. Declare it from the user's own words, BEFORE
            # the guard below, which would otherwise read the reference as a
            # hallucinated type and coerce it to String — losing the closed
            # value set.
            self._declare_referenced_enumerations(
                system_spec, raw_request or user_request)

            # Guard: never ship an attribute whose TYPE names another class.
            # The deterministic code generators only handle primitive + enum
            # attribute types; a class-typed attribute crashes them and forces
            # the expensive LLM-from-scratch fallback. Rewrite it into an
            # association (or coerce an unknown type to String).
            self._rewrite_class_typed_attributes(system_spec)

            # Guard: a label naming BOTH ends ("contact / bookingsAsContact")
            # becomes a name plus a sourceRole. A name with a space fails the
            # editor's class-diagram quality check and blocks GUI generation.
            # Must run before the merge, which uses the recovered source role
            # to spot one fact stated from both sides.
            self._split_dual_role_names(system_spec)

            # Guard: an association class is attached to ONE association; it
            # must not ALSO be joined to that association's endpoints by
            # ordinary associations. SQLAlchemyGenerator derives the link
            # table's FK columns from the ATTACHED association's end names
            # ("bookings" -> bookings_id) and the ordinary association's
            # relationship() from its OWN end name ("booking" -> booking_id),
            # so the two disagree and the generated sql_alchemy.py raises
            # AttributeError on import - the app is dead before the agent
            # writes a line. The system prompt already forbids these extra
            # links; this makes it deterministic.
            self._drop_redundant_association_class_links(system_spec)

            # Guard: one fact stated twice becomes one relationship. A request
            # phrased from both sides ("a booking produces a bill" / "a bill is
            # raised against a booking") makes the LLM emit two links for the
            # same association, which lands as two foreign keys to the same
            # partner and a create schema demanding two ids for one link.
            # Must run before the naming/orientation guards below, which would
            # otherwise paper over the duplicate with a "_1" suffix.
            self._merge_redundant_parallel_associations(system_spec)

            # Guard: no two associations may share the same name. The LLM
            # sometimes names multiple relationships identically (e.g. two links
            # both called "task" in a task-management model) — ambiguous on the
            # canvas and a hard collision when the domain model is built (BUML
            # association names must be unique). Rename duplicates deterministically.
            self._dedupe_relationship_names(system_spec)

            # Guard: no class may end up with two association ends of the same
            # name once this spec is injected (the recurring "cannot have two
            # association ends with the same name" validation error on rich
            # models). Reorients/renames colliding relationships using the
            # exact end-name derivation the validator applies.
            self._ensure_unique_association_ends(system_spec)

            # Guard: an abstract base declared as the CHILD of its own
            # subclasses. Must run before the shadow-strip below, which walks
            # this graph and would otherwise strip from the wrong end.
            self._fix_inverted_inheritance(system_spec)

            # Guard: isAbstract on a LEAF rather than the base. An abstract leaf
            # cannot be instantiated, so the app cannot create that entity at all.
            self._fix_misplaced_abstract(system_spec)

            # Guard: a subclass must not redefine an attribute an ancestor
            # already declares (BUML validates attribute shadowing; the LLM
            # loves stamping id/createdAt/updatedAt on EVERY class, including
            # subclasses of a base that already has them). Strip the shadowed
            # copies deterministically. Method OVERRIDES are left alone —
            # overriding is legitimate OO.
            self._strip_shadowed_attributes(system_spec)

            # Guard: BUML rejects decorated type tokens the LLM sometimes
            # emits ('str?', 'int?'). Normalize them: on an attribute the '?'
            # means optional (strip it, set isOptional); on a parameter or
            # return type it is stripped outright — the metamodel has no
            # optional parameters.
            self._sanitize_member_types(system_spec)

            # Guard: no set of classes may each need the other to exist first.
            # An "at least one" bound facing a link's mandatory back-reference
            # (Booking 1..* BookedRoom, BookedRoom 1 Booking) makes both create
            # schemas demand the other's id, so the generated API can construct
            # neither. Relax one end per cycle and keep the rule as an OCL
            # invariant. Runs after the orientation/naming guards so the
            # invariant navigates the end by its final name.
            self._break_mandatory_cycles(system_spec)

            # Drop any OCL constraint whose context isn't a real class in the
            # spec (the LLM occasionally references a class it didn't create).
            self._validate_constraints(system_spec)

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

            # The matching frontend converter persists constraints as OCL
            # boxes/links and associationClass attachments as ClassLinkRel.
            # Keep this wire contract aligned with websocket_protocol.rst.
            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": self.get_diagram_type(),
                "message": message
            }

        except LLMPredictionError as exc:
            logger.error(f"❌ [ClassDiagram] generate_complete_system LLM FAILED: {exc}")
            return self._incremental_system_fallback(user_request, existing_model, raw_request=raw_request)
        except Exception as exc:
            logger.error(f"❌ [ClassDiagram] generate_complete_system FAILED: {exc}", exc_info=True)
            return self._incremental_system_fallback(user_request, existing_model, raw_request=raw_request)

    # ------------------------------------------------------------------
    # Enum-relationship guard
    # ------------------------------------------------------------------

    @staticmethod
    def _to_camel_case(name: str) -> str:
        """Lowercase the first character so an enum's PascalCase name becomes a
        sensible camelCase attribute name (TaskStatus -> taskStatus)."""
        if not name:
            return name
        return name[0].lower() + name[1:]

    def _rewrite_enum_relationships(self, system_spec: Dict[str, Any]) -> None:
        """Remove relationships whose endpoint is an enumeration, converting each
        into an enum-typed attribute on the non-enum class.

        A class must never have a relationship (Association, Composition, etc.)
        to an Enumeration — the enum belongs as an attribute *type*. The LLM
        occasionally emits such a relationship anyway (e.g. Task --> TaskStatus
        instead of Task.status : TaskStatus). This guard rewrites those into an
        attribute on the non-enum side typed as the enum's name. When neither
        side is a real (non-enum) class, the relationship is dropped.

        Mutates *system_spec* in place. No-op when there are no enum endpoints.
        """
        classes = system_spec.get("classes")
        relationships = system_spec.get("relationships")
        if not isinstance(classes, list) or not isinstance(relationships, list):
            return

        # Map class name -> spec, and the set of enum class names.
        class_by_name: Dict[str, Dict[str, Any]] = {}
        enum_names: set[str] = set()
        for cls in classes:
            if not isinstance(cls, dict):
                continue
            name = cls.get("className")
            if not isinstance(name, str) or not name:
                continue
            class_by_name[name] = cls
            if cls.get("isEnumeration"):
                enum_names.add(name)

        if not enum_names:
            return

        kept: List[Dict[str, Any]] = []
        rewritten = 0
        dropped = 0
        for rel in relationships:
            if not isinstance(rel, dict):
                kept.append(rel)
                continue
            source = rel.get("source")
            target = rel.get("target")
            src_is_enum = source in enum_names
            tgt_is_enum = target in enum_names

            if not src_is_enum and not tgt_is_enum:
                kept.append(rel)
                continue

            # The enum endpoint and the (hopefully non-enum) other endpoint.
            enum_name = source if src_is_enum else target
            other_name = target if src_is_enum else source

            owner = class_by_name.get(other_name)
            # Only attach an attribute when the other side is a real, non-enum
            # class. enum<->enum (or unknown) relationships are simply dropped.
            if owner is not None and other_name not in enum_names:
                attrs = owner.setdefault("attributes", [])
                if not isinstance(attrs, list):
                    attrs = []
                    owner["attributes"] = attrs
                attr_name = self._to_camel_case(enum_name)
                already = any(
                    isinstance(a, dict) and a.get("type") == enum_name
                    for a in attrs
                )
                if not already:
                    attrs.append({
                        "name": attr_name,
                        "type": enum_name,
                        "visibility": "public",
                    })
                rewritten += 1
                logger.info(
                    "[ClassDiagram] Rewrote relationship-to-enum %s<->%s into "
                    "attribute %s.%s : %s",
                    source, target, other_name, attr_name, enum_name,
                )
            else:
                dropped += 1
                logger.info(
                    "[ClassDiagram] Dropped relationship-to-enum %s<->%s "
                    "(no non-enum class to attach an attribute to)",
                    source, target,
                )

        if rewritten or dropped:
            system_spec["relationships"] = kept
            logger.info(
                "[ClassDiagram] Enum-relationship guard: rewrote %d, dropped %d",
                rewritten, dropped,
            )

    def _fix_inverted_inheritance(self, system_spec: Dict[str, Any]) -> None:
        """Flip an Inheritance whose ends are the wrong way round.

        E.g. ``Person -> Guest`` plus ``Person -> Employee``: the converter reads
        source as the SUBCLASS, so this means "Person extends Guest" AND "Person
        extends Employee". DomainModel.validate() only checks that both ends
        exist, so the inverted hierarchy would reach code generation.

        The signal is the SHARED SOURCE, not the abstract flag. A base extended
        by two kinds appears as two links sharing a TARGET; the inverted form
        shares a SOURCE, which would be multiple inheritance -- never what these
        specs ask for. ``isAbstract`` travels independently of the direction (a
        correct ``Employee -> Person`` may still mark Employee abstract), so it
        cannot arbitrate.

        A single inverted link with no shared end is left alone -- there is no
        evidence to act on.
        """
        rels = [r for r in system_spec.get("relationships", [])
                if isinstance(r, dict) and r.get("type") == "Inheritance"]
        if len(rels) < 2:
            return
        names = {c.get("className") for c in system_spec.get("classes", [])
                 if isinstance(c, dict)}

        by_source: Dict[str, list] = {}
        for rel in rels:
            child, parent = rel.get("source"), rel.get("target")
            if child in names and parent in names and child != parent:
                by_source.setdefault(child, []).append(rel)

        flipped = []
        for child, group in by_source.items():
            if len(group) < 2:
                continue
            for rel in group:
                rel["source"], rel["target"] = rel["target"], rel["source"]
                flipped.append(f"{child}->{rel['source']}")

        if flipped:
            logger.info(
                "[ClassDiagram] Flipped %d inverted inheritance link(s) "
                "(one class declared as the subclass of several): %s",
                len(flipped), ", ".join(flipped)[:200],
            )

    def _fix_misplaced_abstract(self, system_spec: Dict[str, Any]) -> None:
        """Clear ``isAbstract`` from a class that is only ever a SUBCLASS.

        The LLM sometimes marks a leaf (e.g. ``Guest``) abstract instead of the
        base. An abstract leaf cannot be instantiated, so the delivered app
        could not create that entity, and BUML does not object: it happily
        holds an abstract class with no subclasses.

        One-directional on purpose. A class that is the SOURCE of an inheritance
        and never the TARGET has nothing beneath it, so abstract is meaningless
        there and clearing it is safe. Promoting the base instead would be a
        guess: an abstract base is not required by these specs.

        Runs after the direction guard, which must settle who is the subclass.
        """
        classes = {
            c.get("className"): c
            for c in system_spec.get("classes", [])
            if isinstance(c, dict) and c.get("className")
        }
        inh = [r for r in system_spec.get("relationships", [])
               if isinstance(r, dict) and r.get("type") == "Inheritance"]
        if not inh or not classes:
            return

        bases = {r.get("target") for r in inh}
        cleared = []
        for rel in inh:
            child = classes.get(rel.get("source"))
            name = rel.get("source")
            if child and name not in bases and child.get("isAbstract"):
                child["isAbstract"] = False
                cleared.append(name)

        if cleared:
            logger.info(
                "[ClassDiagram] Cleared isAbstract from %d leaf class(es) "
                "(a subclass with nothing beneath it cannot be abstract): %s",
                len(cleared), ", ".join(sorted(set(cleared)))[:200],
            )

    def _strip_shadowed_attributes(self, system_spec: Dict[str, Any]) -> None:
        """Remove subclass attributes whose name an ancestor already defines.

        Walks Inheritance relationships (source = subclass, target =
        superclass) transitively, so multi-level chains are covered and a
        cycle cannot loop. Purely deterministic — no LLM round-trip.
        """
        classes = {
            c.get("className"): c
            for c in system_spec.get("classes", [])
            if isinstance(c, dict) and c.get("className")
        }
        parent_of: Dict[str, str] = {}
        for rel in system_spec.get("relationships", []):
            if not isinstance(rel, dict) or rel.get("type") != "Inheritance":
                continue
            child, parent = rel.get("source"), rel.get("target")
            if child in classes and parent in classes and child != parent:
                parent_of[child] = parent

        if not parent_of:
            return

        # Snapshot BEFORE any stripping: inherited sets must come from the
        # original declarations, not from classes already mutated this pass
        # (matters for chains where an ancestor is itself stripped, and for
        # degenerate LLM-emitted inheritance cycles).
        original_attrs = {
            name: {
                a.get("name")
                for a in cls.get("attributes", [])
                if isinstance(a, dict) and a.get("name")
            }
            for name, cls in classes.items()
        }

        def _ancestor_attribute_names(name: str) -> set:
            names: set = set()
            seen: set = {name}
            parent = parent_of.get(name)
            while parent and parent not in seen:
                seen.add(parent)
                names.update(original_attrs.get(parent, set()))
                parent = parent_of.get(parent)
            return names

        for name, cls in classes.items():
            inherited = _ancestor_attribute_names(name)
            if not inherited:
                continue
            attributes = cls.get("attributes", [])
            kept = [
                a for a in attributes
                if not (isinstance(a, dict) and a.get("name") in inherited)
            ]
            if len(kept) != len(attributes):
                logger.info(
                    "[ClassDiagram] Stripped %d shadowed attribute(s) from "
                    "'%s' — already defined in an ancestor",
                    len(attributes) - len(kept), name,
                )
                cls["attributes"] = kept

    @staticmethod
    def _to_identifier(raw: str, *, pascal: bool) -> str:
        """Coerce an arbitrary string into a safe identifier.

        Splits on any run of non-alphanumeric characters and rejoins the parts.
        ``pascal=True`` PascalCases every part (for class/enum names);
        ``pascal=False`` keeps the first part's case and camelCases the rest
        (for attribute/method/parameter names). A leading digit is prefixed with
        an underscore. Returns ``""`` when *raw* has no usable characters, so
        the caller can decide to leave the original alone.
        """
        parts = [p for p in re.split(r"[^A-Za-z0-9]+", raw or "") if p]
        if not parts:
            return ""
        if pascal:
            ident = "".join(p[:1].upper() + p[1:] for p in parts)
        else:
            ident = parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])
        if ident and ident[0].isdigit():
            ident = "_" + ident
        return ident

    def _sanitize_identifier_names(self, system_spec: Dict[str, Any]) -> None:
        """Rewrite class/enum/member names that aren't valid identifiers.

        A class named with hyphens or spaces (e.g. ``risk-awareness-recommendation``)
        is rejected by BUML validation, and the editor's auto-fix loop can never
        repair it — it tries to *rename* a class that was never accepted, spins,
        and burns turns. Fixing the name here, before the
        spec is injected, removes the failure at its source.

        Class/enum names are PascalCased and every reference to them — relationship
        ``source``/``target``, class-typed attribute ``type``, and OCL constraint
        ``context`` — is rewritten with the same mapping so the spec stays
        internally consistent. Attribute/method/parameter names are fixed in place
        (nothing references them by name). Mutates *system_spec* in place.
        """
        classes = system_spec.get("classes")
        rename_map: Dict[str, str] = {}

        def _needs_fix(value: Any) -> bool:
            return (
                isinstance(value, str)
                and value.strip() != ""
                and _VALID_IDENTIFIER_RE.match(value.strip()) is None
            )

        if isinstance(classes, list):
            for cls in classes:
                if not isinstance(cls, dict):
                    continue
                old = cls.get("className")
                if _needs_fix(old):
                    new = self._to_identifier(old, pascal=True)
                    if new and new != old:
                        cls["className"] = new
                        rename_map[old] = new
                for attr in cls.get("attributes", []) or []:
                    if isinstance(attr, dict) and _needs_fix(attr.get("name")):
                        fixed = self._to_identifier(attr["name"], pascal=False)
                        if fixed:
                            attr["name"] = fixed
                for method in cls.get("methods", []) or []:
                    if not isinstance(method, dict):
                        continue
                    if _needs_fix(method.get("name")):
                        fixed = self._to_identifier(method["name"], pascal=False)
                        if fixed:
                            method["name"] = fixed
                    for param in method.get("parameters", []) or []:
                        if isinstance(param, dict) and _needs_fix(param.get("name")):
                            fixed = self._to_identifier(param["name"], pascal=False)
                            if fixed:
                                param["name"] = fixed

        if not rename_map:
            return

        relationships = system_spec.get("relationships")
        if isinstance(relationships, list):
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                for key in ("source", "target", "associationClass"):
                    if rel.get(key) in rename_map:
                        rel[key] = rename_map[rel[key]]

        if isinstance(classes, list):
            for cls in classes:
                if not isinstance(cls, dict):
                    continue
                for attr in cls.get("attributes", []) or []:
                    if isinstance(attr, dict) and attr.get("type") in rename_map:
                        attr["type"] = rename_map[attr["type"]]

        constraints = system_spec.get("constraints")
        if isinstance(constraints, list):
            for con in constraints:
                if isinstance(con, dict) and con.get("context") in rename_map:
                    con["context"] = rename_map[con["context"]]

        logger.info(
            "[ClassDiagram] Sanitized %d invalid identifier name(s): %s",
            len(rename_map),
            ", ".join(f"{k!r}->{v!r}" for k, v in rename_map.items())[:200],
        )

    def _sanitize_member_types(self, system_spec: Dict[str, Any]) -> None:
        """Normalize decorated type tokens BUML would reject (e.g. 'str?').

        The LLM sometimes carries the optional marker into the type itself
        ('description: str?'). On attributes '?' has a real meaning — strip it
        and set isOptional. On method parameters and return types the
        metamodel has no optionality, so the marker is just stripped.
        Purely deterministic — no LLM round-trip.

        Also coerces a method parameter or return type that names nothing in
        the spec ('List[Bill]', 'dict', 'Money'). Attributes already get that
        treatment in :meth:`_rewrite_class_typed_attributes`; methods did not,
        and the omission is not local: ``parse_method`` raises ValueError and
        ``_resolve_type`` raises ConversionError, so ONE bad token aborts the
        conversion of the WHOLE diagram rather than just that method.
        """
        fixed = 0
        coerced = 0

        known: set = set()
        for cls in system_spec.get("classes", []):
            if isinstance(cls, dict) and isinstance(cls.get("className"), str):
                known.add(cls["className"])

        def _clean(token: Any) -> Any:
            nonlocal fixed
            if isinstance(token, str) and token.rstrip().endswith("?"):
                fixed += 1
                return token.rstrip().rstrip("?").strip()
            return token

        def _resolvable(token: Any, fallback: str) -> Any:
            """Keep a primitive/class/enum name; replace anything else."""
            nonlocal coerced
            if not isinstance(token, str) or not token.strip():
                return token
            name = token.strip()
            alias = self._TYPE_ALIAS_NORMALISATION.get(name.lower())
            if alias is not None:
                return alias
            if name.lower() in self._PRIMITIVE_ATTR_TYPES or name in known:
                return name
            coerced += 1
            logger.info("[ClassDiagram] Unresolvable member type %r -> %r",
                        name, fallback)
            return fallback

        for cls in system_spec.get("classes", []):
            if not isinstance(cls, dict):
                continue
            for attr in cls.get("attributes", []):
                if not isinstance(attr, dict):
                    continue
                cleaned = _clean(attr.get("type"))
                if cleaned != attr.get("type"):
                    attr["type"] = cleaned or None
                    attr["isOptional"] = True
            for method in cls.get("methods", []):
                if not isinstance(method, dict):
                    continue
                # 'any' for a return: non-committal and a valid primitive, the
                # same landing spot 'void' already gets downstream.
                method["returnType"] = _resolvable(
                    _clean(method.get("returnType")), "any")
                for param in method.get("parameters", []):
                    if isinstance(param, dict):
                        param["type"] = _resolvable(_clean(param.get("type")), "str")
        if fixed:
            logger.info(
                "[ClassDiagram] Sanitized %d decorated type token(s) "
                "('type?' → 'type')", fixed,
            )
        if coerced:
            logger.info(
                "[ClassDiagram] Coerced %d unresolvable method member type(s); "
                "an unknown token would have failed the whole conversion", coerced,
            )

    def _dedupe_relationship_names(self, system_spec: Dict[str, Any]) -> None:
        """Ensure no two relationships share the same name.

        The LLM occasionally gives several associations the same name (e.g. two
        different links both called "task" in a task-management model). Duplicate
        names are ambiguous on the canvas and collide when the domain model is
        built (BUML association names must be unique). Rename each duplicate
        deterministically — preferring a meaningful ``sourceTarget`` camelCase
        name, falling back to a numeric suffix. Unnamed associations are left
        alone (the editor auto-labels them). Mutates *system_spec* in place.
        """
        relationships = system_spec.get("relationships")
        if not isinstance(relationships, list):
            return

        def _norm(value: Any) -> str:
            return value.strip().lower() if isinstance(value, str) else ""

        seen: set = set()
        renamed = 0
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            raw = rel.get("name")
            key = _norm(raw)
            if not key:
                continue  # unnamed — the editor auto-labels it, no collision
            if key not in seen:
                seen.add(key)
                continue
            base = raw.strip() if isinstance(raw, str) else "relationship"
            source = rel.get("source") if isinstance(rel.get("source"), str) else ""
            target = rel.get("target") if isinstance(rel.get("target"), str) else ""
            unique = ""
            if source and target:
                derived = source[:1].lower() + source[1:] + target[:1].upper() + target[1:]
                if _norm(derived) and _norm(derived) not in seen:
                    unique = derived
            if not unique:
                idx = 2
                while _norm(f"{base}{idx}") in seen:
                    idx += 1
                unique = f"{base}{idx}"
            rel["name"] = unique
            seen.add(_norm(unique))
            renamed += 1
            logger.info(
                "[ClassDiagram] Renamed duplicate relationship '%s' -> '%s'",
                base, unique,
            )

        if renamed:
            logger.info(
                "[ClassDiagram] Deduped %d duplicate relationship name(s)", renamed
            )

    # ------------------------------------------------------------------
    # Association-end uniqueness at CREATION time (spec-level guard)
    # ------------------------------------------------------------------
    # The frontend converter (ClassDiagramConverter.convertCompleteSystem)
    # injects every spec relationship with ``source.role = rel.sourceRole ||
    # ''`` and ``target.role = rel.name || ''``, and the validator derives
    # each end's name as "explicit role, else lowercased endpoint-class
    # name". So for a spec relationship S→T:
    #   * S's navigable end (the target endpoint) is named rel.name or
    #     lower(T);
    #   * T's navigable end (the source endpoint) is named rel.sourceRole or
    #     lower(S).
    # BOTH ends are nameable, so a collision is always fixable in place —
    # never by swapping the endpoints, which would move each name to the
    # opposite end.

    # Spec relationship types that create NO association ends once injected
    # (the converter maps them to ClassInheritance). Everything else —
    # including Realization/Dependency, which fall through to
    # ClassBidirectional — produces two role-bearing endpoints.
    _SPEC_NON_END_REL_TYPES = frozenset({"inheritance", "generalization"})

    @staticmethod
    def _spec_role(rel: Dict[str, Any], side: str) -> str:
        """The explicit role a spec relationship gives its *side* end.

        ``name`` carries the TARGET end's role (the converter writes it to
        ``target.role``), ``sourceRole`` the SOURCE end's. Empty when the end
        is unnamed, in which case the converter falls back to the lowercased
        endpoint-class name.
        """
        raw = rel.get("name") if side == "target" else rel.get("sourceRole")
        return raw.strip() if isinstance(raw, str) and raw.strip() else ""

    @staticmethod
    def _spec_rel_type(rel: Dict[str, Any]) -> str:
        value = rel.get("type")
        return value.strip().lower() if isinstance(value, str) else "association"

    _SPEC_UNBOUNDED = 9999

    @classmethod
    def _parse_multiplicity(cls, value: Any) -> tuple:
        """``'1'`` -> (1, 1); ``'0..1'`` -> (0, 1); ``'0..*'`` -> (0, UNBOUNDED).

        Unparseable input falls back to (1, 1) — the schema default.
        """
        text = str(value).strip().replace(" ", "") if value is not None else ""
        if not text:
            return (1, 1)
        star = cls._SPEC_UNBOUNDED

        def _bound(token: str, default: int) -> int:
            if token in ("*", "n", "many"):
                return star
            try:
                return int(token)
            except ValueError:
                return default

        if ".." in text:
            low, _, high = text.partition("..")
            return (_bound(low, 0), _bound(high, star))
        single = _bound(text, 1)
        return (single, single)

    @classmethod
    def _format_multiplicity(cls, lower: int, upper: int) -> str:
        if lower == upper:
            return "*" if upper >= cls._SPEC_UNBOUNDED else str(lower)
        high = "*" if upper >= cls._SPEC_UNBOUNDED else str(upper)
        return f"{lower}..{high}"

    def _split_dual_role_names(self, system_spec: Dict[str, Any]) -> None:
        """Split a ``"targetRole / sourceRole"`` label into its two roles.

        Asked to state both directions of a relationship, the model volunteers
        both role names in the single label field — ``"contact /
        bookingsAsContact"``. That reaches the editor as an association-end
        name, and a name containing a space fails the class-diagram quality
        check outright ("Name cannot contain spaces"), blocking GUI
        generation.

        The second name is real information the schema never asked for, so it
        is kept as ``sourceRole`` (which the orientation guard already swaps
        alongside multiplicities) rather than thrown away — the duplicate
        detector uses it to recognise one fact stated from both sides. Any
        label that is not a valid identifier is coerced into one.

        Mutates *system_spec* in place.
        """
        relationships = system_spec.get("relationships")
        if not isinstance(relationships, list):
            return
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            label = rel.get("name")
            if not isinstance(label, str) or not label.strip():
                continue
            target_role, _, source_role = label.partition("/")
            target_ident = self._to_identifier(target_role, pascal=False)
            if target_ident:
                rel["name"] = target_ident
            elif _VALID_IDENTIFIER_RE.match(label.strip()) is None:
                rel.pop("name", None)
            if source_role.strip() and not rel.get("sourceRole"):
                source_ident = self._to_identifier(source_role, pascal=False)
                if source_ident:
                    rel["sourceRole"] = source_ident
            if label != rel.get("name"):
                logger.info(
                    "[ClassDiagram] Split relationship label %r into name=%r "
                    "sourceRole=%r", label, rel.get("name"), rel.get("sourceRole"),
                )

    @classmethod
    def _is_crosswise_reciprocal(cls, group: List[Dict[str, Any]]) -> bool:
        """True when two opposite-facing links carry each other's role names.

        ``Room--reservations/room-->ReservedRoom`` beside
        ``ReservedRoom--room/reservations-->Room`` is one fact written from
        each side: each link's target role is the other's source role. That
        holds even when the two statements disagree on a bound (``1`` vs
        ``1..*``), which is exactly when the mirror test fails and the
        duplicate would otherwise survive.
        """
        if len(group) != 2:
            return False
        first, second = group
        if first["source"] != second["target"] or first["target"] != second["source"]:
            return False

        def roles(rel):
            name = rel.get("name")
            source = rel.get("sourceRole")
            if not isinstance(name, str) or not isinstance(source, str):
                return None
            return (name.strip().lower(), source.strip().lower())

        a, b = roles(first), roles(second)
        return bool(a and b and a == (b[1], b[0]))

    @staticmethod
    def _is_default_end_name(name: Any, target: Any) -> bool:
        """True when a label says nothing a missing label would not.

        The converter names an unlabelled end after its class (``booking`` on
        a link to ``Booking``), so a label that merely repeats the target
        class — singular or plural, with or without the ``_N`` collision
        suffix — carries no role information and cannot claim a distinct role.
        """
        if not isinstance(name, str) or not isinstance(target, str):
            return False
        base = re.sub(r"_\d+$", "", name.strip().lower())
        cls = target.strip().lower()
        return bool(cls) and base in (cls, cls + "s", cls + "es")

    def _drop_redundant_association_class_links(self, system_spec: Dict[str, Any]) -> None:
        """Remove ordinary links between an association class and its endpoints.

        An association class is ALREADY connected to both endpoints through
        the association it is attached to. The LLM frequently adds the two
        ordinary associations as well, and the result does not merely
        duplicate a link: the generated ``sql_alchemy.py`` names the same
        foreign key two different ways and fails to import. With these two
        links removed it imports.

        Mutates *system_spec* in place.
        """
        relationships = system_spec.get("relationships")
        if not isinstance(relationships, list):
            return

        # {association class name: {the two endpoint class names}}
        attached: Dict[str, set] = {}
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            name = rel.get("associationClass")
            if isinstance(name, str) and name.strip():
                attached[name.strip()] = {rel.get("source"), rel.get("target")}
        if not attached:
            return

        kept, dropped = [], []
        for rel in relationships:
            if not isinstance(rel, dict):
                kept.append(rel)
                continue
            source, target = rel.get("source"), rel.get("target")
            # Association, Composition and Aggregation alike: the LLM picks
            # any of the three for these duplicates, and all three produce
            # the same unimportable sql_alchemy.py. Inheritance is excluded -
            # "the link class IS a Booking" is a different claim, wrong in
            # other ways, and not this guard's business.
            redundant = (
                str(rel.get("type") or "Association")
                in ("Association", "Composition", "Aggregation")
                and not rel.get("associationClass")
                and ((source in attached and target in attached[source])
                     or (target in attached and source in attached[target]))
            )
            (dropped if redundant else kept).append(rel)

        if dropped:
            system_spec["relationships"] = kept
            logger.info(
                "[ClassDiagram] Dropped %d redundant association-class link(s): %s",
                len(dropped),
                ", ".join(f"{r.get('source')}->{r.get('target')}" for r in dropped)[:200],
            )

    def _merge_redundant_parallel_associations(self, system_spec: Dict[str, Any]) -> None:
        """Collapse an association the LLM emitted more than once.

        The generator restates one fact as two links whenever the request
        phrases it from both sides ("a booking produces a bill" / "a bill is
        raised against a booking"). Both survive into the model, so the class
        gets two foreign keys to the same partner and the create schema demands
        two ids for one relationship (e.g. ``Bill`` carrying both ``booking_id``
        and ``forBooking_id``, each NOT NULL and UNIQUE).

        Genuinely distinct parallel links (``homeAddress``/``workAddress``) are
        told apart by their names: a group whose relationships all carry
        DISTINCT non-empty names is left alone. Any other group — one named and
        one unnamed, all unnamed, or repeated names — is one fact stated twice
        and is merged into a single relationship. A label that merely repeats
        its target class (``booking`` on a link to ``Booking``) is what the
        converter derives for a missing label, so it counts as unnamed here;
        otherwise one fact named after its own target from both sides slips
        past this test and surfaces downstream as a ``_1``-suffixed duplicate
        end and a second foreign key.

        The survivor keeps the first occurrence's orientation and the first
        non-empty name; each end's multiplicity becomes the union of the
        group's (min lower bound, max upper bound), so the merged link admits
        every instance either original admitted. Only plain Associations are
        grouped: Composition/Aggregation carry existential semantics that must
        not be dissolved, and Inheritance creates no ends.

        Mutates *system_spec* in place; no-op for already-clean specs.
        """
        relationships = system_spec.get("relationships")
        if not isinstance(relationships, list) or len(relationships) < 2:
            return

        groups: Dict[frozenset, List[Dict[str, Any]]] = {}
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            if self._spec_rel_type(rel) != "association":
                continue
            # An explicit attachment is semantic identity, not a duplicate
            # inferred from endpoints/names. Never merge it into a plain link
            # (or another attributed link) and silently lose its attributes.
            if rel.get("associationClass"):
                continue
            src, tgt = rel.get("source"), rel.get("target")
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            if not src or not tgt or src == tgt:
                continue
            groups.setdefault(frozenset((src, tgt)), []).append(rel)

        doomed: List[int] = []
        for pair, group in groups.items():
            if len(group) < 2:
                continue
            names = [
                rel["name"].strip()
                for rel in group
                if isinstance(rel.get("name"), str) and rel["name"].strip()
                and not self._is_default_end_name(rel["name"], rel["target"])
            ]
            all_distinctly_named = (
                len(names) == len(group) and len(set(names)) == len(group)
            )
            # All distinctly named => genuinely different roles, keep them —
            # unless the group is a RECIPROCAL pair whose ends mirror each
            # other. A plain Association is already navigable both ways, so
            # A->B plus B->A with mirrored multiplicities is one fact the
            # request stated from each side, each side named after its own
            # target (e.g. Booking--bookingRooms-->BookingRoom alongside
            # BookingRoom--booking-->Booking).
            if all_distinctly_named and not self._is_crosswise_reciprocal(group):
                # Kept on purpose, but say so: EF Core and SQLAlchemy both
                # surface this ambiguity rather than resolving it silently, and
                # downstream the extra link shows up only as a "_1" suffix that
                # reads like a bug.
                logger.info(
                    "[ClassDiagram] Kept %d parallel %s-%s associations (%s) — "
                    "distinct role names, so they are treated as different "
                    "relationships, not one stated twice",
                    len(group), *sorted(pair), ", ".join(names),
                )
                continue

            survivor = group[0]
            keep_src, keep_tgt = survivor["source"], survivor["target"]
            src_low, src_up = self._parse_multiplicity(survivor.get("sourceMultiplicity"))
            tgt_low, tgt_up = self._parse_multiplicity(survivor.get("targetMultiplicity"))

            for rel in group[1:]:
                a_low, a_up = self._parse_multiplicity(rel.get("sourceMultiplicity"))
                b_low, b_up = self._parse_multiplicity(rel.get("targetMultiplicity"))
                if rel["source"] != keep_src:
                    # Stated in the opposite direction — line the ends up.
                    a_low, a_up, b_low, b_up = b_low, b_up, a_low, a_up
                src_low, src_up = min(src_low, a_low), max(src_up, a_up)
                tgt_low, tgt_up = min(tgt_low, b_low), max(tgt_up, b_up)
                doomed.append(id(rel))

            survivor["sourceMultiplicity"] = self._format_multiplicity(src_low, src_up)
            survivor["targetMultiplicity"] = self._format_multiplicity(tgt_low, tgt_up)
            if not (isinstance(survivor.get("name"), str) and survivor["name"].strip()):
                if names:
                    survivor["name"] = names[0]
            logger.info(
                "[ClassDiagram] Merged %d duplicate %s-%s associations into one "
                "(%s %s -- %s %s)",
                len(group), keep_src, keep_tgt,
                keep_src, survivor["sourceMultiplicity"],
                survivor["targetMultiplicity"], keep_tgt,
            )

        if doomed:
            dead = set(doomed)
            system_spec["relationships"] = [
                rel for rel in relationships
                if not (isinstance(rel, dict) and id(rel) in dead)
            ]

    @staticmethod
    def _reach_over_edges(edges: List[tuple], seeds: set) -> set:
        """Seeds plus everything reachable following the directed edges BOTH
        ways (ancestors ∪ descendants — never siblings), matching the
        duplicate-end repair's inheritance-closure semantics."""
        def _reach(forward: bool) -> set:
            reached = set(seeds)
            frontier = set(seeds)
            while frontier:
                nxt = set()
                for a, b in edges:
                    if forward and a in frontier and b not in reached:
                        nxt.add(b)
                    elif not forward and b in frontier and a not in reached:
                        nxt.add(a)
                reached |= nxt
                frontier = nxt
            return reached

        return _reach(True) | _reach(False)

    def _ensure_unique_association_ends(self, system_spec: Dict[str, Any]) -> None:
        """Make every class's association-end names unique in the spec.

        Computes each class's navigable end names with the same derivation
        the validator uses on the injected model (explicit role — ``name`` on
        the target endpoint, ``sourceRole`` on the source endpoint — else the
        lowercased endpoint-class name), counting ends inherited across the
        class's generalization chain. A duplicate group keeps its first
        entry, preferring an LLM-given name over a class-derived one so an
        OCL-navigable role is never the one renamed; every other end is
        suffixed ``_1``/``_2`` — the convention the backend converter and the
        repair flow use — writing ``name`` for a target end and
        ``sourceRole`` for a source end. New target names also stay unique
        among relationship labels.

        Endpoints are NEVER swapped: the wire format carries the target role
        in ``name`` and the source role in ``sourceRole``, so a swap moves
        each name to the opposite end. Both ends being nameable, naming is
        always enough.

        Mutates *system_spec* in place; no-op for already-clean specs.
        """
        relationships = system_spec.get("relationships")
        if not isinstance(relationships, list) or not relationships:
            return
        classes = system_spec.get("classes")

        def _is_end_creating(rel: Any) -> bool:
            return (
                isinstance(rel, dict)
                and self._spec_rel_type(rel) not in self._SPEC_NON_END_REL_TYPES
                and isinstance(rel.get("source"), str) and rel.get("source")
                and isinstance(rel.get("target"), str) and rel.get("target")
            )

        # Inheritance closure: an end counts against every class in the chain.
        inheritance_edges = [
            (rel["source"], rel["target"])
            for rel in relationships
            if isinstance(rel, dict)
            and self._spec_rel_type(rel) in self._SPEC_NON_END_REL_TYPES
            and isinstance(rel.get("source"), str)
            and isinstance(rel.get("target"), str)
        ]

        def _collect_entries() -> List[Dict[str, Any]]:
            """One entry per (relationship, owning class) in document order."""
            entries: List[Dict[str, Any]] = []
            for rel in relationships:
                if not _is_end_creating(rel):
                    continue
                for side, owner in (("target", rel["source"]),
                                    ("source", rel["target"])):
                    explicit = self._spec_role(rel, side)
                    entries.append({
                        "rel": rel, "owner": owner, "side": side,
                        "explicit": bool(explicit),
                        "name": explicit or rel[side].lower(),
                    })
            return entries

        taken_labels = {
            rel["name"].strip().lower()
            for rel in relationships
            if isinstance(rel, dict) and isinstance(rel.get("name"), str)
            and rel["name"].strip()
        }

        class_order: List[str] = []
        if isinstance(classes, list):
            for cls in classes:
                if isinstance(cls, dict) and isinstance(cls.get("className"), str):
                    class_order.append(cls["className"])
        for entry in _collect_entries():
            if entry["owner"] not in class_order:
                class_order.append(entry["owner"])

        renamed = 0
        for cls_name in class_order:
            entries = _collect_entries()  # recompute after prior mutations
            chain = self._reach_over_edges(inheritance_edges, {cls_name})
            owned = [e for e in entries if e["owner"] in chain]
            taken_ends = {e["name"] for e in owned}
            by_name: Dict[str, List[Dict[str, Any]]] = {}
            for e in owned:
                by_name.setdefault(e["name"], []).append(e)
            for end_name, group in by_name.items():
                if len(group) < 2:
                    continue
                explicit = [e for e in group if e["explicit"]]
                kept = (explicit or group)[0]
                for e in group:
                    if e is kept:
                        continue
                    # Both ends are nameable: a target end through ``name``, a
                    # source end through ``sourceRole`` (the converter writes
                    # both). The collision once treated as unfixable comes from
                    # DIFFERENT pairs meeting through inheritance —
                    # Booking->Person[contact] and Booking->Employee[handledBy]
                    # each give their target an end called "booking", and
                    # Employee inherits Person's.
                    rel, side = e["rel"], e["side"]
                    field = "name" if side == "target" else "sourceRole"
                    base = self._spec_role(rel, side) or rel[side].lower()
                    counter = 1
                    while (
                        f"{base}_{counter}" in taken_ends
                        or (side == "target"
                            and f"{base}_{counter}".lower() in taken_labels)
                    ):
                        counter += 1
                    unique = f"{base}_{counter}"
                    rel[field] = unique
                    taken_ends.add(unique)
                    if side == "target":
                        taken_labels.add(unique.lower())
                    renamed += 1
                    logger.info(
                        "[ClassDiagram] Named the %s end of the %s-%s "
                        "association '%s' so '%s' has no duplicate '%s' end",
                        side, rel["source"], rel["target"], unique, cls_name,
                        end_name,
                    )

        if renamed:
            logger.info(
                "[ClassDiagram] Assigned %d unique association-end name(s)",
                renamed,
            )

    # Attribute types the deterministic generators accept without an association.
    _PRIMITIVE_ATTR_TYPES = frozenset({
        "string", "str", "int", "integer", "long", "short", "bool", "boolean",
        "float", "double", "decimal", "number", "char", "byte",
        "date", "datetime", "time", "timedelta", "any",
    })

    # Types the frontend already canonicalises before anything reaches BUML
    # (``typeNormalization.ts`` TYPE_ALIASES). The agent did not, so a
    # ``LocalDate`` matched no primitive, was not a declared class, and fell
    # to the coercion below as an unresolvable type - **a date silently
    # became a string**. Same for ``BigDecimal``, 3 occurrences each across
    # 19 runs. Mapping rather than merely admitting them matters: the
    # membership test returns the name unchanged, which would hand BUML a
    # ``LocalDate`` it does not know.
    _TYPE_ALIAS_NORMALISATION = {
        "localdate": "date",
        "localdatetime": "datetime",
        "localtime": "time",
        "timestamp": "datetime",
        "instant": "datetime",
        "bigdecimal": "float",
        "biginteger": "int",
        "uuid": "str",
        "guid": "str",
        "character": "str",
        "object": "any",
        "void": "any",
    }

    def _declare_referenced_enumerations(
        self, system_spec: Dict[str, Any], request_text: str,
    ) -> None:
        """Declare an enumeration an attribute references but nobody created.

        :meth:`_rewrite_class_typed_attributes` coerces an attribute whose
        type names nothing in the spec to ``String``. For a genuinely
        hallucinated type that is right. For ``Ticket.status : StatusEnum``
        it throws away the one thing the model got right.

        With Qwen3-30B-A3B, ``String``-typed attributes are in practice exactly
        these closed states: ordinary attributes are lowercase
        ``str``/``int``/``date``, because ``String`` is not a token the model
        writes here, it is what the coercion below assigns. The log shows it::

            Coerced unknown attribute type Ticket.status : StatusEnum -> String
            Coerced unknown attribute type Ticket.urgency : UrgencyEnum -> String

        The enumerations were missing from the model's ``classes`` list, not
        from its intent. Recover them instead of erasing them.

        Detection is deterministic. Only the literal NAMES need language - a
        specification says "the guests have not arrived yet", not
        NOT_ARRIVED - so they come from one narrow structured call, made only
        when there is something to recover (no candidates, no call, so a
        model that declares its enumerations pays nothing). The repair fails
        closed: whatever is not recovered is left untouched for the existing
        guard to coerce exactly as before.
        """
        classes = system_spec.get("classes")
        if not isinstance(classes, list) or not request_text:
            return

        declared = {
            cls["className"] for cls in classes
            if isinstance(cls, dict) and isinstance(cls.get("className"), str)
        }
        candidates: Dict[str, List[str]] = {}
        for cls in classes:
            if not isinstance(cls, dict) or cls.get("isEnumeration"):
                continue
            for attr in cls.get("attributes") or []:
                if not isinstance(attr, dict):
                    continue
                type_name = attr.get("type")
                if not isinstance(type_name, str) or not type_name.strip():
                    continue
                type_name = type_name.strip()
                if (type_name.lower() in self._PRIMITIVE_ATTR_TYPES
                        or type_name in declared
                        # A type that is not already a valid identifier would
                        # become an invalid CLASS name here, and
                        # _sanitize_identifier_names has already run. Leave it
                        # to the coercion below, as today.
                        or _VALID_IDENTIFIER_RE.match(type_name) is None
                        # Same population the coercion below calls hallucinated:
                        # a PascalCase type naming nothing in the spec. No name
                        # pattern beyond that — "DayOfWeek" is as much an
                        # enumeration as "OrderStatus", and the recovery call is
                        # what decides, answering "no fixed set" for a genuine
                        # hallucination.
                        or not type_name[:1].isupper()):
                    continue
                candidates.setdefault(type_name, []).append(
                    f"{cls.get('className')}.{attr.get('name')}")
        if not candidates:
            return

        wanted = "\n".join(
            f"- {type_name} (referenced by {', '.join(sorted(owners))})"
            for type_name, owners in sorted(candidates.items())
        )
        prompt = (
            "These attribute types are referenced by the model of this "
            "system but were never declared:\n"
            f"{wanted}\n\n"
            "For each one, read the specification below and give the values "
            "that attribute is restricted to, in the order the specification "
            "lists them, as UPPER_SNAKE_CASE identifiers. Name each value "
            "after what the specification says it means: 'the guests have "
            "not arrived yet' is NOT_ARRIVED, 'awaiting payment' is "
            "AWAITING_PAYMENT. If the specification does not restrict an "
            "attribute to a fixed set of values, return an EMPTY list for "
            "it - never invent values.\n\n"
            f"Specification:\n{request_text}"
        )
        try:
            recovered = self.predict_structured(
                prompt,
                RecoveredEnumerationsSpec,
                system_prompt=(
                    "You extract enumerated value sets from a system "
                    "specification. You never invent a value the "
                    "specification does not state."
                ),
                model=MODEL_GENERATION_SMALL,
            )
        except Exception as exc:  # noqa: BLE001 - recovery is best-effort
            logger.warning(
                "[ClassDiagram] Enumeration recovery failed (%s); leaving %d "
                "undeclared type(s) to the class-typed-attribute guard",
                exc, len(candidates),
            )
            return

        added = 0
        for entry in recovered.enumerations:
            type_name = (entry.typeName or "").strip()
            if type_name not in candidates or type_name in declared:
                continue
            literals: List[str] = []
            seen: set = set()
            for literal in entry.literals:
                # UPPER_SNAKE, keeping the underscores: _to_identifier() would
                # camelCase "NOT_ARRIVED" into "notArrived" and lose them.
                name = re.sub(r"[^A-Za-z0-9]+", "_", str(literal or "")).strip("_").upper()
                if name[:1].isdigit():
                    name = "_" + name
                if not name or name in seen:
                    continue
                seen.add(name)
                literals.append(name)
            if len(literals) < 2:
                # One value is not a choice, and none means the specification
                # closed nothing. Either way leave it to the existing guard.
                continue
            classes.append({
                "className": type_name,
                "attributes": [
                    {"name": literal, "type": None, "visibility": "public",
                     "isDerived": False, "defaultValue": None,
                     "isOptional": False, "isExternalId": False}
                    for literal in literals
                ],
                "methods": [],
                "isAbstract": False,
                "isEnumeration": True,
            })
            declared.add(type_name)
            added += 1
            logger.info(
                "[ClassDiagram] Declared enumeration %s(%s) referenced by %s",
                type_name, ", ".join(literals), ", ".join(candidates[type_name]),
            )
        if added:
            logger.info(
                "[ClassDiagram] Undeclared-enumeration guard: %d of %d recovered",
                added, len(candidates),
            )

    def _rewrite_class_typed_attributes(self, system_spec: Dict[str, Any]) -> None:
        """Convert an attribute TYPED as another class into an Association.

        The deterministic FastAPI/SQLAlchemy/Pydantic generators accept only
        primitive + enum attribute types. A class-typed attribute (e.g.
        ``PointOfInterest.location : Location``) raises "unsupported attribute
        type" and forces the expensive LLM-from-scratch rebuild. This guard
        rewrites such an attribute into an ``Association`` owner->target (role
        = the attribute name, so ``startLocation``/``endLocation`` stay
        distinct). A PascalCase type naming no class/enum in the spec is a
        hallucinated reference and is coerced to ``String``.

        The INVERSE guard (relationship-to-enum -> attribute) is
        :meth:`_rewrite_enum_relationships`. Mutates *system_spec* in place;
        no-op when every attribute is already a primitive/enum.
        """
        classes = system_spec.get("classes")
        if not isinstance(classes, list):
            return

        class_names: set[str] = set()
        enum_names: set[str] = set()
        for cls in classes:
            if not isinstance(cls, dict):
                continue
            name = cls.get("className")
            if isinstance(name, str) and name:
                class_names.add(name)
                if cls.get("isEnumeration"):
                    enum_names.add(name)
        non_enum_classes = class_names - enum_names

        relationships = system_spec.get("relationships")
        if not isinstance(relationships, list):
            relationships = []
            system_spec["relationships"] = relationships

        converted = 0
        coerced = 0
        for cls in classes:
            if not isinstance(cls, dict) or cls.get("isEnumeration"):
                continue
            owner = cls.get("className")
            attrs = cls.get("attributes")
            if not owner or not isinstance(attrs, list):
                continue
            kept: List[Dict[str, Any]] = []
            for a in attrs:
                if not isinstance(a, dict):
                    kept.append(a)
                    continue
                t = a.get("type")
                if not isinstance(t, str) or not t.strip():
                    kept.append(a)
                    continue
                tt = t.strip()
                alias = self._TYPE_ALIAS_NORMALISATION.get(tt.lower())
                if alias is not None:
                    # A Java-flavoured spelling the frontend already
                    # canonicalises. Without this it matches no primitive,
                    # is not a declared class, and falls to the coercion
                    # below - so a LocalDate silently became a String.
                    a["type"] = alias
                    kept.append(a)
                    continue
                if tt.lower() in self._PRIMITIVE_ATTR_TYPES or tt in enum_names:
                    kept.append(a)  # primitive or valid enum-typed attribute
                    continue
                if tt in non_enum_classes:
                    # class-typed attribute -> association owner -> target
                    relationships.append({
                        "type": "Association",
                        "source": owner,
                        "target": tt,
                        "sourceMultiplicity": "0..*",
                        "targetMultiplicity": "0..1" if a.get("isOptional") else "1",
                        "name": a.get("name") or self._to_camel_case(tt),
                    })
                    converted += 1
                    logger.info(
                        "[ClassDiagram] Rewrote class-typed attribute %s.%s : %s "
                        "into an Association", owner, a.get("name"), tt,
                    )
                    # drop the attribute (replaced by the association)
                else:
                    # PascalCase but not a known class/enum/primitive -> a
                    # hallucinated type reference; keep the field, make it a String.
                    a["type"] = "String"
                    coerced += 1
                    logger.info(
                        "[ClassDiagram] Coerced unknown attribute type %s.%s : %s "
                        "-> String", owner, a.get("name"), tt,
                    )
                    kept.append(a)
            cls["attributes"] = kept

        if converted or coerced:
            logger.info(
                "[ClassDiagram] Class-typed-attribute guard: %d -> association, "
                "%d -> String", converted, coerced,
            )

    def _break_mandatory_cycles(self, system_spec: Dict[str, Any]) -> None:
        """Relax one end of every mandatory creation cycle, keeping the rule as OCL.

        The LLM renders "a booking covers at least one room" as
        ``Booking --[1..*]--> BookedRoom`` while the link class keeps
        ``BookedRoom --[1]--> Booking``. Each class then needs an instance of
        the other to exist first, so the generated ``BookingCreate`` demands a
        ``BookedRoom`` id, ``BookedRoomCreate`` a ``Booking`` id, and the API
        can construct neither. BESSER's
        ``DomainModel._validate_mandatory_cycles`` reports the shape and the
        Spec-Driven Agent refuses to build on it.

        "At least one" is a business invariant, not a creation-time bound:
        ``0..*`` plus ``context Booking inv: self.bookedRooms->size() >= 1``
        states the same rule at the same strength, checked on a complete
        object instead of at insert time. Dependencies are derived as the
        validator derives them — an end with ``min >= 1`` means the class at
        the other end needs this end's type, and a mandatory self-association
        cycles on its own. Per cycle ONE end is relaxed, chosen by: a
        many-valued end (``1..*``) over a single-valued one (``1``), which
        keeps the child's mandatory parent reference — the NOT NULL foreign
        key — intact; then the target end over the source end (the spec's
        named, navigable end, so a composition keeps its whole); then document
        order. Repeats until nothing cycles. The invariant navigates the end
        by the name the converter will give it (``name`` on a target end,
        ``sourceRole`` on a source end, else the lowercased endpoint class)
        because B-OCL resolves it against the context class.

        Mutates *system_spec* in place; no-op when nothing cycles.
        """
        relationships = system_spec.get("relationships")
        if not isinstance(relationships, list) or not relationships:
            return
        order = {id(rel): i for i, rel in enumerate(relationships)}

        def _mandatory_ends() -> List[Dict[str, Any]]:
            """One entry per end with min >= 1: the class at the OTHER end
            (``needer``) cannot be created without this end's type."""
            ends: List[Dict[str, Any]] = []
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                if self._spec_rel_type(rel) in self._SPEC_NON_END_REL_TYPES:
                    continue
                src, tgt = rel.get("source"), rel.get("target")
                if not isinstance(src, str) or not isinstance(tgt, str):
                    continue
                if not src or not tgt:
                    continue
                for side, needer, needed in (("target", src, tgt), ("source", tgt, src)):
                    low, up = self._parse_multiplicity(rel.get(f"{side}Multiplicity"))
                    if low >= self._SPEC_UNBOUNDED:
                        low = 0  # a bare "*" parses as (star, star); it means 0..*
                    if low >= 1:
                        ends.append({
                            "rel": rel, "side": side, "needer": needer,
                            "needed": needed, "min": low, "max": up,
                        })
            return ends

        def _find_cycle(ends: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
            needs: Dict[str, List[Dict[str, Any]]] = {}
            for end in ends:
                needs.setdefault(end["needer"], []).append(end)
            for start in needs:
                stack: List[tuple] = [(start, [])]
                seen: set = set()
                while stack:
                    node, path = stack.pop()
                    on_path = [start] + [e["needed"] for e in path]
                    for end in needs.get(node, ()):
                        if end["needed"] in on_path:
                            return path[on_path.index(end["needed"]):] + [end]
                        if end["needed"] not in seen:
                            seen.add(end["needed"])
                            stack.append((end["needed"], path + [end]))
            return None

        constraints = system_spec.get("constraints")
        if not isinstance(constraints, list):
            constraints = []
        taken_names: set = set()
        for con in constraints:
            if not isinstance(con, dict):
                continue
            if isinstance(con.get("name"), str):
                taken_names.add(con["name"].strip())
            match = re.search(r"\binv\s+(\w+)\s*:", str(con.get("expression") or ""))
            if match:
                taken_names.add(match.group(1))

        relaxed = 0
        for _ in range(2 * len(relationships)):  # each pass drops one mandatory end
            cycle = _find_cycle(_mandatory_ends())
            if not cycle:
                break
            victim = min(cycle, key=lambda e: (
                e["max"] <= 1, e["side"] != "target", order[id(e["rel"])],
            ))
            rel, side = victim["rel"], victim["side"]
            before = rel.get(f"{side}Multiplicity")
            rel[f"{side}Multiplicity"] = self._format_multiplicity(0, victim["max"])
            end_name = self._spec_role(rel, side) or rel[side].lower()
            base = f"{end_name}_at_least_{victim['min']}"
            name, counter = base, 2
            while name in taken_names:
                name = f"{base}_{counter}"
                counter += 1
            taken_names.add(name)
            constraints.append({
                "context": victim["needer"],
                "expression": (
                    f"context {victim['needer']} inv {name}: "
                    f"self.{end_name}->size() >= {victim['min']}"
                ),
                "name": name,
            })
            relaxed += 1
            logger.info(
                "[ClassDiagram] Relaxed the %s end of %s-%s from %s to %s to break "
                "the mandatory creation cycle %s; kept the rule as OCL invariant '%s'",
                side, rel["source"], rel["target"], before,
                rel[f"{side}Multiplicity"],
                " -> ".join([cycle[0]["needer"]] + [e["needed"] for e in cycle]),
                name,
            )

        if relaxed:
            system_spec["constraints"] = constraints
            logger.info("[ClassDiagram] Broke %d mandatory creation cycle(s)", relaxed)

    def _validate_constraints(self, system_spec: Dict[str, Any]) -> None:
        """Drop OCL constraints whose context isn't a real class in the spec.

        Keeps only constraints with a non-empty ``expression`` whose ``context``
        names a class that actually exists, so a hallucinated context can't reach
        the editor. Mutates *system_spec* in place. No-op when there are none.
        """
        constraints = system_spec.get("constraints")
        if not isinstance(constraints, list) or not constraints:
            return
        class_names = {
            c.get("className")
            for c in system_spec.get("classes", [])
            if isinstance(c, dict) and c.get("className")
        }
        kept: List[Dict[str, Any]] = []
        for con in constraints:
            if not isinstance(con, dict):
                continue
            expr = con.get("expression")
            context = con.get("context")
            if not isinstance(expr, str) or not expr.strip():
                continue
            if context not in class_names:
                logger.info(
                    "[ClassDiagram] Dropped OCL constraint with unknown context %r",
                    context,
                )
                continue
            kept.append(con)
        system_spec["constraints"] = kept
        if kept:
            logger.info("[ClassDiagram] Captured %d OCL constraint(s)", len(kept))

    @staticmethod
    def _enum_names_in_model(current_model: Optional[Dict[str, Any]]) -> set[str]:
        """Return the set of enumeration class names present in *current_model*.

        Enumerations are stored in the editor model as elements with
        ``type == "Enumeration"``.
        """
        names: set[str] = set()
        if not isinstance(current_model, dict):
            return names
        elements = current_model.get("elements")
        if not isinstance(elements, dict):
            return names
        for el in elements.values():
            if isinstance(el, dict) and el.get("type") == "Enumeration":
                name = el.get("name")
                if isinstance(name, str) and name.strip():
                    names.add(name.strip())
        return names

    def _normalise_modification_types(self, spec: Dict[str, Any]) -> None:
        """Canonicalise Java-flavoured type names on the MODIFY path.

        Every type guard — ``_sanitize_member_types``,
        ``_rewrite_class_typed_attributes``, ``_declare_referenced_enumerations``
        — is called from ``generate_complete_system`` only, so a type arriving
        through a modification reached BUML with nothing looking at it. The
        same ``LocalDate`` that the generate path canonicalises to ``date``
        went through a modify op untouched.

        This is the deterministic half of that gap and it is safe to close
        here: the mapping mirrors the frontend's own ``TYPE_ALIASES``, so a
        modify op and a full generation now agree. The other half — coercing a
        genuinely unknown type, and rewriting a class-typed attribute into an
        association — needs the declared-type set and is left to the caller
        rather than guessed at from one op.

        Mutates *spec* in place. No-op when no op carries an aliased type.
        """
        mods = spec.get("modifications")
        if not isinstance(mods, list):
            mods = [spec]

        def _canon(value: Any) -> Any:
            if not isinstance(value, str) or not value.strip():
                return value
            return self._TYPE_ALIAS_NORMALISATION.get(value.strip().lower(), value)

        renamed = 0
        for mod in mods:
            if not isinstance(mod, dict):
                continue
            changes = mod.get("changes")
            if not isinstance(changes, dict):
                continue
            for key in ("type", "returnType"):
                before = changes.get(key)
                after = _canon(before)
                if after != before:
                    changes[key] = after
                    renamed += 1
            for list_key, type_key in (("attributes", "type"),
                                       ("methods", "returnType"),
                                       ("parameters", "type")):
                for entry in changes.get(list_key) or []:
                    if not isinstance(entry, dict):
                        continue
                    before = entry.get(type_key)
                    after = _canon(before)
                    if after != before:
                        entry[type_key] = after
                        renamed += 1

        if renamed:
            logger.info(
                "[ClassDiagram] Canonicalised %d aliased type(s) on the "
                "modification path", renamed,
            )

    def _rewrite_enum_relationship_mods(
        self, spec: Dict[str, Any], current_model: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Rewrite add_relationship mods that point at an enumeration.

        A relationship endpoint that is an enumeration — whether the enum
        already exists in *current_model* or is created by an add_class with
        ``isEnumeration=true`` earlier in the same batch — is converted into an
        ``add_attribute`` on the non-enum class, typed as the enum's name. If
        neither endpoint is a real class, the relationship mod is dropped.

        Mutates *spec* in place. No-op when no relationship touches an enum.
        """
        # Normalize to the flat inner-mod list.
        if isinstance(spec.get("modifications"), list):
            mods = spec["modifications"]
            is_batch = True
        elif isinstance(spec.get("modification"), dict):
            mods = [spec["modification"]]
            is_batch = False
        else:
            return

        # Enum names from the existing model plus any enum being added now.
        enum_names = self._enum_names_in_model(current_model)
        for m in mods:
            if not isinstance(m, dict) or m.get("action") != "add_class":
                continue
            changes = m.get("changes") or {}
            if isinstance(changes, dict) and changes.get("isEnumeration"):
                cn = changes.get("className")
                if isinstance(cn, str) and cn.strip():
                    enum_names.add(cn.strip())

        if not enum_names:
            return

        new_mods: List[Dict[str, Any]] = []
        changed = False
        for m in mods:
            if not isinstance(m, dict) or m.get("action") != "add_relationship":
                new_mods.append(m)
                continue
            target = m.get("target") or {}
            src = target.get("sourceClass") if isinstance(target, dict) else None
            tgt = target.get("targetClass") if isinstance(target, dict) else None
            src_is_enum = src in enum_names
            tgt_is_enum = tgt in enum_names
            if not src_is_enum and not tgt_is_enum:
                new_mods.append(m)
                continue

            enum_name = src if src_is_enum else tgt
            other_name = tgt if src_is_enum else src
            changed = True
            if other_name and other_name not in enum_names:
                new_mods.append({
                    "action": "add_attribute",
                    "target": {
                        "className": other_name,
                        "attributeName": self._to_camel_case(enum_name),
                    },
                    "changes": {
                        "name": self._to_camel_case(enum_name),
                        "type": enum_name,
                    },
                })
                logger.info(
                    "[ClassDiagram] Rewrote add_relationship %s<->%s into "
                    "add_attribute %s.%s : %s",
                    src, tgt, other_name, self._to_camel_case(enum_name), enum_name,
                )
            else:
                logger.info(
                    "[ClassDiagram] Dropped add_relationship %s<->%s "
                    "(enum endpoint with no non-enum class)",
                    src, tgt,
                )

        if not changed:
            return

        if is_batch:
            spec["modifications"] = new_mods
        else:
            # Was a single modification; promote to batch only if needed.
            spec.pop("modification", None)
            if len(new_mods) == 1:
                spec["modification"] = new_mods[0]
            else:
                spec["modifications"] = new_mods

    # Matches an explicit trailing list of named entities, e.g. "... with
    # members, tiers, transactions, and rewards" or "... including patients,
    # doctors, appointments, prescriptions". Used only as a deterministic,
    # LLM-free last resort — see ``_extract_named_entities_heuristic`` below.
    _ENTITY_LIST_TRIGGER_RE = re.compile(
        r"\b(?:with|including|involving|having|containing)\s+(.+)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _singularize(word: str) -> str:
        """Naive plural -> singular normalization for class naming (Books -> Book)."""
        lower = word.lower()
        if lower.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"
        if lower.endswith(("ses", "xes", "zes", "ches", "shes")):
            return word[:-2]
        if lower.endswith("s") and not lower.endswith("ss"):
            return word[:-1]
        return word

    @classmethod
    def _extract_named_entities_heuristic(cls, text: Optional[str]) -> List[str]:
        """Deterministically pull candidate class names out of an explicit
        entity list in the user's request (e.g. "...with members, tiers,
        transactions, and rewards"). No LLM call involved.

        Used as a last-resort, non-lossy safety net inside
        ``_incremental_system_fallback``: when the LLM-based class-name
        extraction (below) fails or comes back empty, a request that clearly
        names its entities should still yield those classes instead of a
        single generic "Entity" stub. Returns an empty list when no such
        list is found (plain requests like "create a library management
        system" are left untouched).
        """
        if not isinstance(text, str) or not text.strip():
            return []

        match = None
        for m in cls._ENTITY_LIST_TRIGGER_RE.finditer(text):
            match = m  # keep the match closest to the end of the request
        if not match:
            return []

        # Stop at sentence-ending punctuation so trailing clauses (e.g. "...
        # and then generate Django code") don't leak into the entity list.
        fragment = re.split(r"[.!?;]", match.group(1))[0]
        fragment = re.sub(r"\band\b", ",", fragment, flags=re.IGNORECASE).replace("&", ",")

        names: List[str] = []
        seen: set = set()
        for part in fragment.split(","):
            words = re.findall(r"[A-Za-z][A-Za-z0-9]*", part)
            if not words:
                continue
            candidate = cls._singularize(words[-1])
            pascal = candidate[:1].upper() + candidate[1:]
            key = pascal.lower()
            if len(pascal) < 2 or key in seen:
                continue
            seen.add(key)
            names.append(pascal)
        return names[:10]

    def _incremental_system_fallback(
        self,
        user_request: str,
        existing_model: Optional[Dict[str, Any]] = None,
        raw_request: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fallback: try to generate the system by creating classes individually.

        When the full system generation fails, this extracts class names from
        the user's request and generates each one separately, then combines
        them into a system spec.
        """
        logger.info("[ClassDiagram] Attempting incremental fallback generation")

        # Try to extract class names from the request via the LLM first.
        extraction_prompt = (
            "From this request, extract ONLY the class/entity names the user wants. "
            "Return a JSON array of strings. Example: [\"User\", \"Product\", \"Order\"]\n\n"
            f"Request: {user_request}\n\n"
            "Return ONLY the JSON array, no explanations."
        )

        class_names: List[Any] = []
        try:
            response = self.predict_with_retry(extraction_prompt, max_retries=1)
            cleaned = self.clean_json_response(response)
            import json as _json
            parsed_names = _json.loads(cleaned)
            if isinstance(parsed_names, list) and parsed_names:
                class_names = parsed_names
        except Exception as exc:
            logger.warning(f"[ClassDiagram] LLM class-name extraction failed: {exc}")

        if not class_names:
            # Deterministic, LLM-free last resort: pull entity names
            # directly out of an explicit list in the request itself, so a
            # request that clearly names its entities (e.g. "...with members,
            # tiers, transactions, and rewards") never degrades to a single
            # generic "Entity" stub just because the extraction call above
            # also failed/returned malformed JSON.
            class_names = self._extract_named_entities_heuristic(raw_request or user_request)
            if class_names:
                logger.info(
                    f"[ClassDiagram] LLM extraction empty — recovered "
                    f"{len(class_names)} entity name(s) heuristically: {class_names}"
                )

        if not class_names:
            logger.warning("[ClassDiagram] Could not extract any class names, using basic fallback")
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
                f"I had a bit of trouble building everything at once, but I set up "
                f"{len(classes)} item(s): {class_names_str}. "
                "Want me to connect them together?"
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
        parts = [f"Added **{name}**"]
        if attr_names:
            parts.append(f" — it keeps track of {', '.join(f'`{n}`' for n in attr_names)}")
            if len(attrs) > 5:
                parts.append(f" (+{len(attrs) - 5} more)")
        if methods:
            parts.append(f" and can do {len(methods)} thing(s)")
        parts.append(". Want to add more details or connect it to something else?")
        return "".join(parts)

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        """Build a descriptive message for a complete class diagram spec."""
        system_name = spec.get("systemName", "System")
        classes = spec.get("classes", [])
        rels = spec.get("relationships", [])
        class_names = [c.get("className", "?") for c in classes[:5]]
        msg = f"Your **{system_name}** spec is ready"
        if class_names:
            msg += f" — it captures {', '.join(f'**{n}**' for n in class_names)}"
            if len(classes) > 5:
                msg += f" and {len(classes) - 5} more"
            if rels:
                msg += f", with {len(rels)} relationship(s) linking them"
        elif rels:
            msg += f", with {len(rels)} relationship(s) between them"
        constraints = spec.get("constraints") or []
        if constraints:
            # Honest message: the business rules are understood but the editor
            # has no slot to display/store them yet, so don't claim they're
            # shown on the canvas.
            msg += (
                f". I also noted {len(constraints)} rule(s) you mentioned, though "
                "they aren't shown on the canvas yet"
            )
        msg += "."
        return msg

    # ------------------------------------------------------------------
    # Missing-target validation (drop phantom removals/edits)
    # ------------------------------------------------------------------

    # Actions that reference something that must ALREADY exist (removals /
    # in-place edits). Additions are never validated here.
    _MUST_EXIST_ACTIONS = frozenset({
        "remove_element", "modify_class",
        "modify_attribute", "modify_method", "modify_relationship",
    })

    @staticmethod
    def _clean_member_name(raw: Optional[str]) -> str:
        """Reduce a displayed member to its bare name.

        Handles both shapes the editor stores:

            "email: str"                -> "email"      (attribute)
            "+follow(user: User): bool" -> "follow"     (method)

        The parameter list MUST be cut before the ``:`` split: in a method the
        first ``:`` belongs to a PARAMETER, not a return type, so
        ``follow(user: User)`` would become ``"follow(user"`` and never match a
        bare "follow" (every method lookup and method-related auto-fix would
        fail).
        """
        name = (raw or "").strip()
        if name and name[0] in "+-#~":
            name = name[1:].strip()
        if "(" in name:                       # method: everything before the params
            name = name.split("(", 1)[0].strip()
        elif ":" in name:                     # attribute: drop the type suffix
            name = name.split(":", 1)[0].strip()
        return name

    def _build_model_index(
        self, current_model: Optional[Dict[str, Any]],
    ) -> tuple[set, set, set]:
        """Return ``(class_names, attr_names, method_names)`` (all lowercased)
        present in *current_model*, for validating modification targets.

        Handles the Apollon editor format (attributes/methods stored as separate
        ``ClassAttribute``/``ClassMethod`` elements linked to their class by
        ``owner``) AND an inline format (a class element carrying an
        ``attributes``/``methods`` list or dict).
        """
        class_names: set = set()
        attr_names: set = set()
        method_names: set = set()
        if not isinstance(current_model, dict):
            return class_names, attr_names, method_names
        elements = current_model.get("elements")
        if not isinstance(elements, dict):
            return class_names, attr_names, method_names

        def _add_inline_members(container: Dict[str, Any]) -> None:
            for key, bucket in (("attributes", attr_names), ("methods", method_names)):
                members = container.get(key)
                if isinstance(members, dict):
                    members = list(members.values())
                if not isinstance(members, list):
                    continue
                for mem in members:
                    nm = mem.get("name") if isinstance(mem, dict) else (mem if isinstance(mem, str) else None)
                    if isinstance(nm, str) and nm.strip():
                        bucket.add(self._clean_member_name(nm).lower())

        for el in elements.values():
            if not isinstance(el, dict):
                continue
            el_type = el.get("type")
            name = el.get("name")
            if el_type in ("Class", "AbstractClass", "Enumeration") or (
                el_type is None and isinstance(el.get("attributes"), (list, dict))
            ):
                if isinstance(name, str) and name.strip():
                    class_names.add(name.strip().lower())
                _add_inline_members(el)
            elif el_type == "ClassAttribute":
                if isinstance(name, str) and name.strip():
                    attr_names.add(self._clean_member_name(name).lower())
            elif el_type == "ClassMethod":
                if isinstance(name, str) and name.strip():
                    method_names.add(self._clean_member_name(name).lower())

        return class_names, attr_names, method_names

    def _phantom_note_for_mod(
        self, mod: Dict[str, Any], class_names: set, attr_names: set, method_names: set,
    ) -> Optional[str]:
        """Return a plain-language "couldn't find" note when *mod* removes or
        edits a target that doesn't exist in the model, else ``None`` (keep it).

        Conservative: an attribute/method is treated as missing only when it is
        absent from EVERY class (so an op that names the wrong class but a real
        field is still applied), and additions are never validated.
        """
        action = mod.get("action")
        if action not in self._MUST_EXIST_ACTIONS:
            return None
        target = mod.get("target")
        if not isinstance(target, dict):
            return None

        verb = "remove" if action == "remove_element" else "update"
        cn = (target.get("className") or "").strip()
        attr = (target.get("attributeName") or "").strip()
        meth = (target.get("methodName") or "").strip()
        src = (target.get("sourceClass") or "").strip()
        tgt = (target.get("targetClass") or "").strip()

        # 1) Attribute target — missing only when absent everywhere.
        if attr:
            if attr.lower() not in attr_names:
                return f"I couldn't find a **{attr}** field to {verb}."
            return None
        # 2) Method target.
        if meth:
            if meth.lower() not in method_names:
                return f"I couldn't find a **{meth}** method to {verb}."
            return None
        # 3) Relationship target (by endpoints) — missing when a named endpoint
        #    class doesn't exist.
        if src or tgt:
            missing = [c for c in (src, tgt) if c and c.lower() not in class_names]
            if missing:
                if src and tgt:
                    return (
                        f"I couldn't find a connection between **{src}** and "
                        f"**{tgt}** to {verb}."
                    )
                return f"I couldn't find **{missing[0]}** to {verb} a connection for."
            return None
        # 4) Class target (remove a class, rename a class, …).
        if cn:
            if cn.lower() not in class_names:
                return f"I couldn't find **{cn}** to {verb}."
            return None
        return None

    def _drop_phantom_target_ops(
        self, spec: Dict[str, Any], current_model: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Drop removal/modify ops whose named target is absent from the model.

        Returns a list of plain-language "couldn't find …" notes (one per dropped
        op) for the user. Additions are never touched. Mutates *spec* in place,
        rewriting its ``modification``/``modifications`` payload with the ops that
        survived. When every op is dropped the spec is left with no ops (the
        caller surfaces the notes as a plain message instead).
        """
        if not isinstance(spec, dict) or spec.get("action") != "modify_model":
            return []
        elements = current_model.get("elements") if isinstance(current_model, dict) else None
        if not isinstance(elements, dict) or not elements:
            # No model to validate against → don't drop anything.
            return []

        if isinstance(spec.get("modifications"), list):
            mods = spec["modifications"]
        elif isinstance(spec.get("modification"), dict):
            mods = [spec["modification"]]
        else:
            return []

        class_names, attr_names, method_names = self._build_model_index(current_model)

        kept: List[Any] = []
        notes: List[str] = []
        for mod in mods:
            note = (
                self._phantom_note_for_mod(mod, class_names, attr_names, method_names)
                if isinstance(mod, dict) else None
            )
            if note is None:
                kept.append(mod)
            else:
                notes.append(note)
                logger.info(
                    f"[ClassDiagram] Dropped phantom-target op "
                    f"({mod.get('action')}): {note}"
                )

        if not notes:
            return []

        spec.pop("modification", None)
        spec.pop("modifications", None)
        if len(kept) == 1:
            spec["modification"] = kept[0]
        elif kept:
            spec["modifications"] = kept
        return notes

    # ------------------------------------------------------------------
    # Modification Support (Existing - Updated for new architecture)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Duplicate association-end repair (deterministic — no LLM round-trip)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_duplicate_end_errors(user_request: str) -> tuple:
        """Parse duplicate-association-end validation errors out of a request.

        Returns ``(dup_errors, pure)`` where ``dup_errors`` is a list of
        ``(class_name, end_name)`` tuples and ``pure`` is True when the
        request contains no OTHER bulleted validation errors (i.e. the
        whole repair can be handled deterministically without an LLM call).
        """
        text = user_request or ""
        dup_errors = [
            (m.group("cls"), m.group("end"))
            for m in _DUP_END_ERROR_RE.finditer(text)
        ]
        if not dup_errors:
            return [], False
        bullet_lines = [
            line.strip() for line in text.splitlines()
            if line.strip().startswith("- ")
        ]
        other_bullets = [
            line for line in bullet_lines if not _DUP_END_ERROR_RE.search(line)
        ]
        return dup_errors, not other_bullets

    @staticmethod
    def _effective_end_name(endpoint: Dict[str, Any], name_by_id: Dict[str, str]) -> str:
        """The association-end name the validator sees for one JSON endpoint:
        the explicit role, else the endpoint class's lowercased name (the
        backend converter's fallback)."""
        role = endpoint.get("role") if isinstance(endpoint, dict) else None
        if isinstance(role, str) and role.strip():
            return role.strip()
        cls_name = name_by_id.get(endpoint.get("element")) if isinstance(endpoint, dict) else None
        return cls_name.lower() if isinstance(cls_name, str) else ""

    @classmethod
    def _class_ids_by_name(cls, model: Dict[str, Any]) -> Dict[str, set]:
        """Map class name → set of element ids (class-like elements only)."""
        ids: Dict[str, set] = {}
        elements = model.get("elements")
        if not isinstance(elements, dict):
            return ids
        for eid, el in elements.items():
            if not isinstance(el, dict):
                continue
            el_type = el.get("type")
            name = el.get("name")
            if isinstance(name, str) and name and el_type in _CLASS_ELEMENT_TYPES:
                ids.setdefault(name, set()).add(eid)
        return ids

    @classmethod
    def _inheritance_closure(cls, model: Dict[str, Any], seed_ids: set) -> set:
        """Seed ids plus every ancestor/descendant reachable over inheritance
        links (both directions — inherited ends collide across the chain).
        Siblings are excluded: chains are followed one direction at a time."""
        relationships = model.get("relationships")
        if not isinstance(relationships, dict):
            return set(seed_ids)
        edges = []
        for rel in relationships.values():
            if not isinstance(rel, dict) or rel.get("type") not in _INHERITANCE_REL_TYPES:
                continue
            src = (rel.get("source") or {}).get("element")
            tgt = (rel.get("target") or {}).get("element")
            if src and tgt:
                edges.append((src, tgt))

        def _reach(seeds: set, forward: bool) -> set:
            reached = set(seeds)
            frontier = set(seeds)
            while frontier:
                nxt = set()
                for a, b in edges:
                    if forward and a in frontier and b not in reached:
                        nxt.add(b)
                    elif not forward and b in frontier and a not in reached:
                        nxt.add(a)
                reached |= nxt
                frontier = nxt
            return reached

        return _reach(set(seed_ids), True) | _reach(set(seed_ids), False)

    def _collect_owned_ends(
        self, model: Dict[str, Any], owner_ids: set, name_by_id: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Association ends OWNED by the given class ids, in document order.

        A class navigates via the end on the OPPOSITE side of each
        association it participates in, so when the class is the JSON
        ``source`` its end is the ``target`` endpoint and vice versa
        (a self-association contributes both)."""
        ends: List[Dict[str, Any]] = []
        relationships = model.get("relationships")
        if not isinstance(relationships, dict):
            return ends
        for rel_id, rel in relationships.items():
            if not isinstance(rel, dict) or rel.get("type") not in _ASSOCIATION_END_REL_TYPES:
                continue
            source = rel.get("source") if isinstance(rel.get("source"), dict) else {}
            target = rel.get("target") if isinstance(rel.get("target"), dict) else {}
            src_el, tgt_el = source.get("element"), target.get("element")
            src_name = name_by_id.get(src_el, "")
            tgt_name = name_by_id.get(tgt_el, "")
            if src_el in owner_ids:
                ends.append({
                    "rel_id": rel_id, "side": "target",
                    "name": self._effective_end_name(target, name_by_id),
                    "source_class": src_name, "target_class": tgt_name,
                    "rel_name": rel.get("name") or "",
                })
            if tgt_el in owner_ids:
                ends.append({
                    "rel_id": rel_id, "side": "source",
                    "name": self._effective_end_name(source, name_by_id),
                    "source_class": src_name, "target_class": tgt_name,
                    "rel_name": rel.get("name") or "",
                })
        return ends

    def _build_duplicate_end_fixes(
        self,
        current_model: Optional[Dict[str, Any]],
        dup_errors: List[tuple],
    ) -> tuple:
        """Build deterministic role-rename modifications for duplicate ends.

        Returns ``(mods, rename_notes, unfixable_notes, dup_pairs)``:

        * ``mods`` — ``modify_relationship`` dicts (addressed by
          ``relationshipId``) whose ``changes.roleName`` gives the duplicate
          end a unique name, in the shape the frontend modifier applies.
        * ``rename_notes`` / ``unfixable_notes`` — user-facing sentences.
        * ``dup_pairs`` — ``frozenset({sourceClass, targetClass})`` keys of
          the duplicated relationships (lowercased), so callers can strip
          conflicting LLM-authored edits of the same relationships.

        Uniqueness is guaranteed by construction: new names are chosen
        against every end name the affected class (and its inheritance
        chain) already navigates, plus names assigned earlier in the batch —
        the same ``name_1`` / ``name_2`` convention the backend converter
        uses. The frontend's ``changes.roleName`` only reaches the TARGET
        endpoint of a relationship, so a duplicate that lives on a JSON
        ``source`` endpoint is reported as unfixable instead of being
        "fixed" with a no-op.
        """
        mods: List[Dict[str, Any]] = []
        rename_notes: List[str] = []
        unfixable_notes: List[str] = []
        dup_pairs: set = set()

        if not isinstance(current_model, dict):
            for cls_name, end_name in dup_errors:
                unfixable_notes.append(
                    f"I couldn't locate the associations of '{cls_name}' to rename "
                    f"the duplicate '{end_name}' end automatically — please rename "
                    f"one of the role names on its associations in the editor "
                    f"(for example to '{end_name}_1')."
                )
            return mods, rename_notes, unfixable_notes, dup_pairs

        ids_by_name = self._class_ids_by_name(current_model)
        name_by_id: Dict[str, str] = {
            eid: name for name, eids in ids_by_name.items() for eid in eids
        }

        for cls_name, end_name in dup_errors:
            seed_ids = ids_by_name.get(cls_name, set())
            if not seed_ids:
                unfixable_notes.append(
                    f"I couldn't find the class '{cls_name}' in the diagram to "
                    f"repair its duplicate '{end_name}' association end."
                )
                continue
            owner_ids = self._inheritance_closure(current_model, seed_ids)
            ends = self._collect_owned_ends(current_model, owner_ids, name_by_id)
            dupes = [e for e in ends if e["name"] == end_name]
            if len(dupes) < 2:
                unfixable_notes.append(
                    f"I couldn't locate two '{end_name}' association ends on "
                    f"'{cls_name}' to repair — please check the role names on its "
                    f"associations in the editor."
                )
                continue

            # Names already in use by this class's ends, plus names assigned
            # earlier in this batch (never introduce a fresh collision).
            taken = {e["name"] for e in ends}
            taken.update(
                m["changes"]["roleName"] for m in mods
                if isinstance(m.get("changes"), dict) and m["changes"].get("roleName")
            )

            # Keep one duplicate as-is; prefer keeping an end the frontend
            # cannot rename (a JSON source endpoint) so every renameable
            # duplicate gets a unique name.
            source_side = [e for e in dupes if e["side"] == "source"]
            kept = source_side[0] if source_side else dupes[0]
            to_rename = [e for e in dupes if e is not kept and e["side"] == "target"]
            leftover = [e for e in dupes if e is not kept and e["side"] == "source"]

            for entry in to_rename:
                counter = 1
                while f"{end_name}_{counter}" in taken:
                    counter += 1
                new_name = f"{end_name}_{counter}"
                taken.add(new_name)
                mods.append({
                    "action": "modify_relationship",
                    "target": {
                        "relationshipId": entry["rel_id"],
                        "sourceClass": entry["source_class"],
                        "targetClass": entry["target_class"],
                    },
                    "changes": {"roleName": new_name},
                })
                dup_pairs.add(frozenset({
                    (entry["source_class"] or "").lower(),
                    (entry["target_class"] or "").lower(),
                }))
                pair_label = (
                    f"{entry['source_class']}–{entry['target_class']}"
                    if entry["source_class"] and entry["target_class"]
                    else cls_name
                )
                rename_notes.append(
                    f"Renamed the duplicate association end '{end_name}' on "
                    f"'{cls_name}' to '{new_name}' (on the {pair_label} "
                    f"association) so each end name is unique."
                )

            if leftover:
                # Deterministic honesty check: the duplicate set could not be
                # fully resolved, so the validation error will persist for the
                # remaining ends — say so instead of claiming success.
                unfixable_notes.append(
                    f"'{cls_name}' still has more than one association end named "
                    f"'{end_name}' that I couldn't rename automatically — please "
                    f"rename one of the role names on its associations in the "
                    f"editor (for example to '{end_name}_{len(mods) + 1}')."
                )

        return mods, rename_notes, unfixable_notes, dup_pairs

    def _repair_duplicate_ends(
        self,
        current_model: Optional[Dict[str, Any]],
        dup_errors: List[tuple],
    ) -> Dict[str, Any]:
        """Fully deterministic repair path for duplicate-end-only requests."""
        mods, rename_notes, unfixable_notes, _pairs = self._build_duplicate_end_fixes(
            current_model, dup_errors,
        )
        if not mods:
            return {
                "action": "assistant_message",
                "message": " ".join(
                    ["I couldn't apply this fix automatically."] + unfixable_notes
                ).strip(),
            }
        spec: Dict[str, Any] = {
            "action": "modify_model",
            "diagramType": self.get_diagram_type(),
            "message": " ".join(rename_notes + unfixable_notes),
        }
        if len(mods) == 1:
            spec["modification"] = mods[0]
        else:
            spec["modifications"] = mods
        logger.info(
            "[ClassDiagram] Deterministic duplicate-end repair: %d rename(s), "
            "%d unfixable", len(mods), len(unfixable_notes),
        )
        return spec

    def _apply_duplicate_end_post_guard(
        self,
        spec: Dict[str, Any],
        dup_errors: List[tuple],
        current_model: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Guarantee end-name uniqueness on a mixed (LLM-handled) repair.

        Strips LLM-authored ``modify_relationship`` edits of the duplicated
        relationships (a label rename does not fix an end-name collision and
        would fight the deterministic rename) and appends the deterministic
        role renames, updating the user-facing message honestly.
        """
        if not isinstance(spec, dict) or spec.get("action") != "modify_model":
            return spec
        mods, rename_notes, unfixable_notes, dup_pairs = self._build_duplicate_end_fixes(
            current_model, dup_errors,
        )
        if not mods and not unfixable_notes:
            return spec

        existing = spec.get("modifications")
        if not isinstance(existing, list):
            single = spec.get("modification")
            existing = [single] if isinstance(single, dict) else []
            spec.pop("modification", None)

        def _touches_dup_pair(mod: Dict[str, Any]) -> bool:
            # Any LLM-authored modify_relationship on a duplicated pair is
            # superseded by the deterministic renames below (label renames are
            # validation no-ops; a second roleName writer would conflict).
            if mod.get("action") != "modify_relationship":
                return False
            target = mod.get("target") if isinstance(mod.get("target"), dict) else {}
            pair = frozenset({
                (target.get("sourceClass") or "").lower(),
                (target.get("targetClass") or "").lower(),
            })
            return pair in dup_pairs

        kept_mods = [
            m for m in existing
            if isinstance(m, dict) and not _touches_dup_pair(m)
        ]
        spec["modifications"] = kept_mods + mods
        if len(spec["modifications"]) == 1:
            spec["modification"] = spec.pop("modifications")[0]

        notes = " ".join(rename_notes + unfixable_notes)
        if notes:
            existing_msg = spec.get("message") or ""
            spec["message"] = f"{existing_msg}\n\n{notes}".strip() if existing_msg else notes
        return spec

    def _ensure_unique_ends_for_added_relationships(
        self,
        spec: Dict[str, Any],
        current_model: Optional[Dict[str, Any]],
    ) -> None:
        """Prevent NEW duplicate association ends from ``add_relationship`` mods.

        The frontend's ``addRelationship`` builds the new relationship with
        ``target.role = changes.name || …`` and ``source.role = ''`` — the
        same shape as complete-system injection — so adding a second link
        between an already-connected pair reproduces the duplicate-end
        validation error. For each add (in batch order, against the current
        model's ends plus earlier adds in the batch):

        * a plain Association whose fixed (source-derived) end would collide
          is FLIPPED to the opposite orientation (endpoints + multiplicities
          swapped);
        * if the flipped source-derived end still collides with an existing
          end that sits on a renameable (target) endpoint, a companion
          ``modify_relationship`` + ``roleName`` mod renames the existing end
          (reusing the repair-flow mechanism);
        * the nameable side then gets a unique ``changes.name`` when needed
          (same ``_1``/``_2`` convention as everywhere else).

        Mutates *spec* in place; silent no-op when nothing would collide.
        """
        if not isinstance(spec, dict) or spec.get("action") != "modify_model":
            return
        mods = spec.get("modifications")
        single = None
        if not isinstance(mods, list):
            single = spec.get("modification")
            mods = [single] if isinstance(single, dict) else []
        adds = [
            m for m in mods
            if isinstance(m, dict) and m.get("action") == "add_relationship"
        ]
        if not adds or not isinstance(current_model, dict):
            return

        ids_by_name = self._class_ids_by_name(current_model)
        name_by_id = {
            eid: name for name, eids in ids_by_name.items() for eid in eids
        }

        def _model_ends(cls_name: str) -> List[Dict[str, Any]]:
            seeds = ids_by_name.get(cls_name, set())
            if not seeds:
                return []
            owners = self._inheritance_closure(current_model, seeds)
            return self._collect_owned_ends(current_model, owners, name_by_id)

        pending: Dict[str, set] = {}
        appended: List[Dict[str, Any]] = []

        def _taken(cls_name: str) -> set:
            return (
                {e["name"] for e in _model_ends(cls_name)}
                | pending.get(cls_name, set())
            )

        for mod in adds:
            target = mod.get("target") if isinstance(mod.get("target"), dict) else {}
            changes = mod.get("changes") if isinstance(mod.get("changes"), dict) else {}
            src = target.get("sourceClass")
            tgt = target.get("targetClass")
            if not (isinstance(src, str) and src and isinstance(tgt, str) and tgt):
                continue
            rel_type = str(changes.get("relationshipType") or "Association").strip().lower()
            if rel_type in ("inheritance", "generalization", "realization", "dependency"):
                continue

            label = ""
            for key_source, key in ((changes, "name"), (changes, "roleName"),
                                    (target, "relationshipName")):
                value = key_source.get(key)
                if isinstance(value, str) and value.strip():
                    label = value.strip()
                    break

            def _repairable(cls_name: str, end_name: str) -> bool:
                """True when every existing end of *cls_name* named *end_name*
                sits on a renameable (target) endpoint — i.e. a companion
                roleName mod can clear the collision."""
                if end_name in pending.get(cls_name, set()):
                    return False  # batch-added fixed ends have no rename lever
                colliding_ends = [
                    e for e in _model_ends(cls_name) if e["name"] == end_name
                ]
                return bool(colliding_ends) and all(
                    e["side"] == "target" for e in colliding_ends
                )

            # Fixed (source-derived) end placement: flip a plain Association
            # when the flipped orientation is clean — or when only the
            # flipped orientation's residual collision is repairable via a
            # companion roleName rename of the existing end.
            if src != tgt and rel_type == "association":
                orig_fixed_collides = src.lower() in _taken(tgt)
                flip_fixed_collides = tgt.lower() in _taken(src)
                should_flip = orig_fixed_collides and (
                    not flip_fixed_collides
                    or (
                        not _repairable(tgt, src.lower())
                        and _repairable(src, tgt.lower())
                    )
                )
                if should_flip:
                    target["sourceClass"], target["targetClass"] = tgt, src
                    # Materialize defaults before swapping so the rendered
                    # multiplicities keep their original meaning.
                    src_mult = changes.get("sourceMultiplicity") or "1"
                    tgt_mult = changes.get("targetMultiplicity") or "*"
                    if not isinstance(mod.get("changes"), dict):
                        mod["changes"] = changes
                    changes["sourceMultiplicity"], changes["targetMultiplicity"] = (
                        tgt_mult, src_mult,
                    )
                    src, tgt = tgt, src
                    logger.info(
                        "[ClassDiagram] Reoriented new %s–%s association to "
                        "avoid a duplicate association end", tgt, src,
                    )

            # Fixed end still colliding: rename the EXISTING colliding end
            # when it sits on a renameable (target) endpoint.
            fixed_end = src.lower()
            if src != tgt and fixed_end in _taken(tgt):
                colliding = [
                    e for e in _model_ends(tgt)
                    if e["name"] == fixed_end and e["side"] == "target"
                ]
                if colliding:
                    existing = colliding[0]
                    taken = _taken(tgt) | {fixed_end}
                    counter = 1
                    while f"{fixed_end}_{counter}" in taken:
                        counter += 1
                    new_role = f"{fixed_end}_{counter}"
                    appended.append({
                        "action": "modify_relationship",
                        "target": {
                            "relationshipId": existing["rel_id"],
                            "sourceClass": existing["source_class"],
                            "targetClass": existing["target_class"],
                        },
                        "changes": {"roleName": new_role},
                    })
                    pending.setdefault(tgt, set()).add(new_role)
                    logger.info(
                        "[ClassDiagram] Renamed existing '%s' end on '%s' to "
                        "'%s' to make room for the new association",
                        fixed_end, tgt, new_role,
                    )
                else:
                    logger.warning(
                        "[ClassDiagram] New %s–%s association leaves '%s' with "
                        "a duplicate '%s' end (not renameable) — left for "
                        "downstream repair", src, tgt, tgt, fixed_end,
                    )

            # Nameable side on the SOURCE class.
            nameable_end = label or tgt.lower()
            if nameable_end in _taken(src):
                base = label or tgt.lower()
                taken = _taken(src) | {nameable_end}
                counter = 1
                while f"{base}_{counter}" in taken:
                    counter += 1
                nameable_end = f"{base}_{counter}"
                if not isinstance(mod.get("changes"), dict):
                    mod["changes"] = changes
                changes["name"] = nameable_end
                logger.info(
                    "[ClassDiagram] Named the new %s–%s association '%s' to "
                    "keep '%s' end names unique", src, tgt, nameable_end, src,
                )
            pending.setdefault(src, set()).add(nameable_end)
            pending.setdefault(tgt, set()).add(fixed_end)

        if appended:
            all_mods = mods + appended
            spec.pop("modification", None)
            spec["modifications"] = all_mods

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate modifications for existing class diagram elements.

        Enhanced with impact analysis: when renaming or removing a class,
        the LLM is informed of dependent relationships so it can cascade
        changes appropriately.
        """
        # Duplicate association-end repair: the metamodel's "two association
        # ends with the same name" error is repaired DETERMINISTICALLY — the
        # LLM historically renamed the relationship label (a validation no-op)
        # and claimed success. When the request carries only this error class,
        # skip the LLM entirely; when it is mixed with other errors, run the
        # normal path and enforce uniqueness in a post-guard below.
        _dup_errors, _dup_pure = self._detect_duplicate_end_errors(user_request)
        if _dup_errors and _dup_pure:
            return self._repair_duplicate_ends(current_model, _dup_errors)
        # Build impact context for modifications that affect relationships
        impact_context = self._build_impact_context(current_model)

        system_prompt = MODIFY_SYSTEM_PROMPT_CLASS

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

        logger.info(f"[ClassDiagram] generate_modification called with: {user_request!r}")
        logger.debug(f"[ClassDiagram] Modification context block length: {len(context_block)} chars")

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
                    # Skip if already has a specific identifier. sourceClass/
                    # targetClass identify a relationship by its endpoints — WITHOUT
                    # them here, "remove the relationship between Order and Customer"
                    # fell through and promoted "Order" to className, deleting the
                    # whole Order class.
                    if any(target.get(k) for k in ("className", "classId",
                                                    "relationshipId", "relationshipName",
                                                    "sourceClass", "targetClass",
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
                """Expand refactoring actions into primitives, then guard against
                any relationship whose endpoint is an enumeration."""
                if handler._is_refactoring_action(spec):
                    logger.info("[ClassDiagram] Detected refactoring action, expanding into primitives")
                    spec = handler._expand_refactoring_actions(spec, current_model)
                handler._rewrite_enum_relationship_mods(spec, current_model)
                # Runs after the expansion, so ops synthesised by a
                # refactoring are canonicalised too.
                handler._normalise_modification_types(spec)
                return spec

            # Up to TWO samples: when EVERY op targets something absent from
            # the model, the parse was almost certainly a sampling glitch
            # (e.g. a remove target that came back as a garbled token; the
            # identical retry succeeds).
            # A fresh sample is cheap on the SMALL tier and turns that class
            # of flake into a non-event; a second total whiff reports
            # honestly as before.
            not_found_notes: List[str] = []
            prior_notes: List[str] = []
            modification_spec: Dict[str, Any] = {}
            for attempt in (1, 2):
                modification_spec = self._execute_modification(
                    user_prompt, system_prompt, ClassModificationResponse,
                    post_processor=_strip_spurious_relationship_mods,
                    spec_processor=_expand_refactoring,
                )
                if modification_spec.get("action") != "modify_model":
                    break  # element-not-found short-circuit etc. — not a whiff

                # Deterministic missing-target validation: drop any removal/
                # edit op whose named target doesn't exist in the model, and
                # tell the user what couldn't be found. Previously these were
                # silently dropped — e.g. "remove priority" on a model with no
                # priority field applied the add half and quietly discarded
                # the remove with zero feedback.
                not_found_notes = self._drop_phantom_target_ops(modification_spec, current_model)
                remaining = (
                    modification_spec.get("modification")
                    or modification_spec.get("modifications")
                )
                if remaining or not not_found_notes:
                    # A retry that came back with nothing at all keeps the
                    # first attempt's diagnosis instead of shipping an empty
                    # modification with no explanation.
                    if not remaining and not not_found_notes and prior_notes:
                        not_found_notes = prior_notes
                    break
                prior_notes = not_found_notes
                if attempt == 1:
                    logger.warning(
                        "[ClassDiagram] Every modification op had a phantom "
                        "target (%s) — retrying once with a fresh sample",
                        " / ".join(not_found_notes)[:200],
                    )

            # If every op was a phantom-target removal/edit, nothing remains to
            # apply — surface the not-found note(s) as a plain message rather
            # than shipping an empty modification.
            if not_found_notes and not (
                modification_spec.get("modification")
                or modification_spec.get("modifications")
            ):
                return {
                    "action": "assistant_message",
                    "message": " ".join(not_found_notes),
                }

            # Replace the generic/default message with a clean, descriptive one
            # for add_class (the name lives in changes.className, not the now-
            # cleared target.className) and for association/linking-class requests.
            self._apply_clean_modification_message(modification_spec, user_request)

            # Append the not-found note(s) so the dropped removal/edit isn't
            # silent even when a legitimate add in the same request succeeded.
            if not_found_notes:
                existing = modification_spec.get("message") or ""
                joined = " ".join(not_found_notes)
                modification_spec["message"] = (
                    f"{existing}\n\n{joined}".strip() if existing else joined
                )

            # Mixed repair request (duplicate-end error alongside other
            # validation errors): the LLM handled the rest — enforce end-name
            # uniqueness deterministically on top of its output.
            if _dup_errors:
                modification_spec = self._apply_duplicate_end_post_guard(
                    modification_spec, _dup_errors, current_model,
                )

            # Prevention: a NEW add_relationship must not introduce a
            # duplicate association end (same derivation as the creation
            # guard; reuses the repair helpers for the current model's ends).
            self._ensure_unique_ends_for_added_relationships(
                modification_spec, current_model,
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
    # Modification message building
    # ------------------------------------------------------------------

    # Case-insensitive markers of a hallucinated placeholder name. Mirrors the
    # schema-level _PLACEHOLDER_TOKENS but lives here as a final defensive guard
    # so no placeholder can ever reach the user-facing success message.
    _PLACEHOLDER_MARKERS = ("placeholder", "classnamehere", "namehere")

    @classmethod
    def _looks_like_placeholder(cls, value: Optional[str]) -> bool:
        if not value or not isinstance(value, str):
            return False
        low = value.lower()
        return any(marker in low for marker in cls._PLACEHOLDER_MARKERS)

    def _apply_clean_modification_message(
        self, spec: Dict[str, Any], user_request: str,
    ) -> None:
        """Set a clean, descriptive success message for add_class / association-class.

        The base ``_execute_modification`` builds the default message from
        ``target.className``, which is intentionally cleared for add_class — so
        the default would read "Added **element**". Here we source the real name
        from ``changes.className`` and, when the user asked for an "association
        class", state that a linking class was created between the two endpoints.

        Mutates *spec* in place. No-op when there is no add_class involved.
        """
        # Normalize to a flat list of inner modifications.
        if isinstance(spec.get("modifications"), list):
            mods = spec["modifications"]
        elif isinstance(spec.get("modification"), dict):
            mods = [spec["modification"]]
        else:
            return

        add_class_mods = [
            m for m in mods
            if isinstance(m, dict) and m.get("action") == "add_class"
        ]
        if not add_class_mods:
            return

        # Resolve clean class names (defensively drop any surviving placeholder).
        new_names: List[str] = []
        for m in add_class_mods:
            changes = m.get("changes") or {}
            name = changes.get("className") if isinstance(changes, dict) else None
            if name and not self._looks_like_placeholder(name):
                new_names.append(name)
        if not new_names:
            return

        wants_assoc = any(
            kw in user_request.lower()
            for kw in ("association class", "associationclass", "link class",
                       "linking class", "junction class", "join class")
        )

        # Association/linking-class request: name the two endpoints if we can
        # find the relationships connecting the new class to existing classes.
        if wants_assoc:
            link_name = new_names[0]
            endpoints = self._endpoints_for_link_class(link_name, mods)
            if len(endpoints) >= 2:
                spec["message"] = (
                    f"Created a linking class **{link_name}** between "
                    f"**{endpoints[0]}** and **{endpoints[1]}** "
                    "(an association class is modeled here as a junction class "
                    "with foreign-key attributes and relationships to both)."
                )
            else:
                spec["message"] = (
                    f"Created a linking class **{link_name}** to associate the "
                    "two classes (modeled as a junction class with foreign-key "
                    "attributes)."
                )
            return

        # Plain add_class — describe what was created using the real name(s).
        if len(new_names) == 1:
            name = new_names[0]
            changes = (add_class_mods[0].get("changes") or {})
            attrs = changes.get("attributes") or []
            attr_names = [a.get("name") for a in attrs if isinstance(a, dict) and a.get("name")][:5]
            msg = f"Added the **{name}** class"
            if attr_names:
                msg += " with attributes: " + ", ".join(f"`{n}`" for n in attr_names)
                if len(attrs) > 5:
                    msg += f" (+{len(attrs) - 5} more)"
            msg += "."
            # When add_class is part of a larger batch, keep the batch summary
            # but lead with the clean class line so the name is never junk.
            if len(mods) > 1:
                msg += f" (applied {len(mods)} changes total)."
            spec["message"] = msg
        else:
            spec["message"] = (
                "Added classes: "
                + ", ".join(f"**{n}**" for n in new_names) + "."
            )

    @staticmethod
    def _endpoints_for_link_class(
        link_name: str, mods: List[Dict[str, Any]],
    ) -> List[str]:
        """Return the existing class names a linking class connects to.

        Scans add_relationship mods for the other end of each relationship that
        touches *link_name* (via target.sourceClass / target.targetClass),
        de-duplicated and excluding the link class itself.
        """
        endpoints: List[str] = []
        for m in mods:
            if not isinstance(m, dict) or m.get("action") != "add_relationship":
                continue
            target = m.get("target") or {}
            if not isinstance(target, dict):
                continue
            src = target.get("sourceClass")
            tgt = target.get("targetClass")
            if link_name in (src, tgt):
                other = tgt if src == link_name else src
                if other and other != link_name and other not in endpoints:
                    endpoints.append(other)
        return endpoints

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

        # 1. Create the enum as a real enumeration. The frontend keys off
        #    changes.isEnumeration; without it we got a plain Class. Enum
        #    literals must NOT be typed by the enum name (that produced
        #    "Low: Status") — leave them type-less.
        enum_attrs = [
            {"name": v, "visibility": "public"}
            for v in values if isinstance(v, str)
        ]
        primitives.append({
            "action": "add_class",
            "target": {"className": enum_name},
            "changes": {
                "className": enum_name,
                "isEnumeration": True,
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
