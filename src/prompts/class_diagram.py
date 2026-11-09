"""Class diagram prompts: design rules, complete-system generation, modification, reasoning and repair."""

from .fragments import (
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
    WHOLE_PART_DIRECTION_RULE,
)


CLASS_ACTIONS_BLOCK = """COMMON ACTIONS:
- add_class — create a NEW class with attributes and methods. Put className, attributes, and methods in "changes".
- modify_class — rename a class or change its properties
- add_attribute / modify_attribute — add or change an attribute on a class
- add_method / modify_method — add or change a method on a class
- add_relationship — create a NEW connection between two classes
- modify_relationship — change an EXISTING relationship (multiplicity, type, name)
- remove_element — delete a class, attribute, method, or relationship

ADVANCED ACTIONS (for structural refactoring):
- extract_class, split_class, merge_classes, promote_attribute, add_enum
- extract_class: target.className = the source class; changes.className = the new class; changes.attributes = the attributes to move (by name). The relationship is added for you.
- split_class: one split_class per new class, with the same fields as extract_class (changes.relationshipType="Inheritance" makes the new classes subclasses of the source).
- merge_classes: target.className = the class merged away (removed); changes.name = the class that remains (existing or new). One merge_classes per class merged away.
- promote_attribute: target.className + target.attributeName = the attribute to turn into a class; changes.className = the new class; optional changes.attributes = its fields. The relationship is added for you.
- add_enum: same fields as add_class with isEnumeration (changes.className, changes.attributes = the values); optional target.className + target.attributeName = an attribute to retype to the enum."""


CLASS_KEY_RULES_BLOCK = f"""KEY RULES:
1. {EXACT_NAMES_RULE}
2. {CHANGES_FIELD_RULE}
3. {REMOVE_ELEMENT_RULE}
4. {MULTI_MOD_ARRAY_RULE}
5. {RENAME_CASCADES_RULE}
6. {DELETE_CLASS_CASCADE_RULE}
7. modify_relationship = update existing. add_relationship = brand new.
8. add_class: set target.className and put className, attributes[], methods[] in "changes".
9. When adding new classes, add relationships connecting them to existing classes. Include multiplicities. This is critical — isolated classes with no relationships are useless.
10. {WHOLE_PART_DIRECTION_RULE}"""


CLASS_EXAMPLES_BLOCK = """Examples:
- "rename User to Customer" → ONE modify_class (no relationship changes needed)
- "add email to User" → add_attribute target.className="User", changes.name="email", changes.type="String" (target.attributeName stays null: the attribute does not exist yet)
- "rename aliases in Brand to altNames" → modify_attribute target.className="Brand", target.attributeName="aliases", changes.name="altNames"
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
    CLASS_ACTIONS_BLOCK,
    OCL_CONSTRAINT_BLOCK,
    MODIFY_CRITICAL_BLOCK,
    NAMING_PASCAL_RULE,
    CLASS_KEY_RULES_BLOCK,
    ENUM_RULES_BLOCK,
    CLASS_EXAMPLES_BLOCK,
    OCL_EXAMPLES_BLOCK,
])


SYSTEM_PROMPT = f"""You are a UML modeling expert. Create a focused class specification based on the user's request.

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


SYSTEM_GENERATION_PROMPT = f"""You are a UML modeling expert. Create a COMPLETE, well-structured class diagram system.

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
6. Connect the classes with meaningful relationships — isolated classes are the most common gap. Use Association (general), Inheritance (is-a, sparingly), Composition (strong has-a), Aggregation (weak has-a), Realization (interface). {WHOLE_PART_DIRECTION_RULE}
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


def reasoning_prompt(user_request: str) -> str:
    return (
        "You are a UML domain modeling expert. Think step by step about "
        "the following system request and plan the class diagram design.\n\n"
        f"User Request: {user_request}\n\n"
        "Analyze:\n"
        "1. What are the core domain entities (classes) needed? Match the "
        "scope to the request: a plain request gets the CORE domain only "
        "(6-10 classes); only plan a bigger model when the user explicitly "
        "asks for a comprehensive/complete system or names many entities.\n"
        "2. What attributes does each class need? (be thorough)\n"
        "3. What relationships connect these classes? What type (Association, "
        "Composition, Aggregation, Inheritance)? For a Composition or "
        "Aggregation say which class is the WHOLE: it becomes the target. "
        "For each one state the two "
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


def enumeration_recovery_prompt(wanted: str, request_text: str) -> str:
    return (
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


ENUMERATION_RECOVERY_SYSTEM_PROMPT = (
    "You extract enumerated value sets from a system "
    "specification. You never invent a value the "
    "specification does not state."
)
