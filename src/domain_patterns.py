"""
Domain Pattern Library
----------------------
Common domain patterns for well-known system types.  When the user's request
matches a known domain, the relevant pattern is injected into the LLM prompt
to dramatically improve the quality of generated class diagrams.

Each pattern contains:
- ``keywords``: terms that trigger this pattern (checked against user message)
- ``synonyms``: alternate terms that map to this pattern
- ``core_classes``: the essential classes with typical attributes
- ``key_relationships``: common relationships between classes
- ``enumerations``: common enumeration types for this domain
- ``design_advice``: domain-specific modeling guidance
- ``notes``: additional modeling tips for the LLM
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS: Dict[str, Dict[str, Any]] = {
    "e-commerce": {
        "keywords": [
            "e-commerce", "ecommerce", "online store", "online shop",
            "shopping", "marketplace", "webshop", "retail",
        ],
        "synonyms": ["shop", "store"],
        "core_classes": [
            {"name": "Customer", "attrs": ["id", "name", "email", "address", "phone"]},
            {"name": "Product", "attrs": ["id", "name", "description", "price", "stock", "category"]},
            {"name": "Order", "attrs": ["id", "orderDate", "status", "totalAmount", "shippingAddress"]},
            {"name": "OrderItem", "attrs": ["quantity", "unitPrice", "subtotal"]},
            {"name": "Payment", "attrs": ["id", "amount", "paymentDate", "method", "status"]},
            {"name": "Category", "attrs": ["id", "name", "description"]},
            {"name": "ShoppingCart", "attrs": ["id", "createdAt"]},
            {"name": "Review", "attrs": ["id", "rating", "comment", "date"]},
        ],
        "key_relationships": [
            ("Customer", "Order", "1", "*", "Association", "places"),
            ("Order", "OrderItem", "1", "1..*", "Composition", "contains"),
            ("OrderItem", "Product", "*", "1", "Association", "references"),
            ("Order", "Payment", "1", "1", "Association", "paidBy"),
            ("Product", "Category", "*", "1", "Association", "belongsTo"),
            ("Customer", "ShoppingCart", "1", "0..1", "Association", "has"),
            ("Customer", "Review", "1", "*", "Association", "writes"),
            ("Review", "Product", "*", "1", "Association", "about"),
        ],
        "enumerations": [
            {"name": "OrderStatus", "values": ["Pending", "Confirmed", "Shipped", "Delivered", "Cancelled"]},
            {"name": "PaymentMethod", "values": ["CreditCard", "DebitCard", "PayPal", "BankTransfer"]},
            {"name": "PaymentStatus", "values": ["Pending", "Completed", "Failed", "Refunded"]},
        ],
        "design_advice": (
            "Separate Order from OrderItem to handle line items properly. "
            "Use OrderStatus and PaymentStatus enumerations for lifecycle tracking. "
            "ShoppingCart is optional but improves UX modeling."
        ),
        "notes": (
            "E-commerce systems need clear separation between Order and OrderItem "
            "(line items). Payment should track method (credit card, PayPal, etc.). "
            "Consider order status lifecycle: Pending -> Confirmed -> Shipped -> Delivered."
        ),
    },
    "library": {
        "keywords": [
            "library", "book", "lending", "borrowing", "catalog",
            "librarian",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Book", "attrs": ["isbn", "title", "publicationYear", "genre", "copies"]},
            {"name": "Author", "attrs": ["id", "name", "biography", "nationality"]},
            {"name": "Member", "attrs": ["id", "name", "email", "membershipDate", "status"]},
            {"name": "Loan", "attrs": ["id", "borrowDate", "dueDate", "returnDate", "status"]},
            {"name": "Category", "attrs": ["id", "name", "description"]},
            {"name": "Librarian", "attrs": ["id", "name", "employeeId", "department"]},
        ],
        "key_relationships": [
            ("Book", "Author", "*", "*", "Association", "writtenBy"),
            ("Member", "Loan", "1", "*", "Association", "borrows"),
            ("Loan", "Book", "*", "1", "Association", "involves"),
            ("Book", "Category", "*", "1", "Association", "classifiedAs"),
            ("Librarian", "Loan", "1", "*", "Association", "manages"),
        ],
        "enumerations": [
            {"name": "LoanStatus", "values": ["Active", "Overdue", "Returned", "Lost"]},
            {"name": "MemberStatus", "values": ["Active", "Suspended", "Expired"]},
            {"name": "Genre", "values": ["Fiction", "NonFiction", "Science", "History", "Biography"]},
        ],
        "design_advice": (
            "Use a many-to-many between Book and Author. Track availability via "
            "copies count minus active loans. Loan is the central transaction entity."
        ),
        "notes": (
            "Library systems should track book availability (copies vs. on loan). "
            "Loans have lifecycle: Active -> Overdue -> Returned. "
            "Books can have multiple authors (many-to-many)."
        ),
    },
    "hospital": {
        "keywords": [
            "hospital", "medical", "healthcare", "clinic", "patient",
            "doctor", "health", "appointment",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Patient", "attrs": ["id", "name", "dateOfBirth", "gender", "phone", "address", "bloodType"]},
            {"name": "Doctor", "attrs": ["id", "name", "specialization", "licenseNumber", "phone"]},
            {"name": "Appointment", "attrs": ["id", "dateTime", "duration", "status", "notes"]},
            {"name": "MedicalRecord", "attrs": ["id", "diagnosis", "treatment", "date", "notes"]},
            {"name": "Department", "attrs": ["id", "name", "floor", "phone"]},
            {"name": "Prescription", "attrs": ["id", "medication", "dosage", "frequency", "startDate", "endDate"]},
            {"name": "Nurse", "attrs": ["id", "name", "shift", "department"]},
        ],
        "key_relationships": [
            ("Patient", "Appointment", "1", "*", "Association", "schedules"),
            ("Doctor", "Appointment", "1", "*", "Association", "attends"),
            ("Patient", "MedicalRecord", "1", "*", "Composition", "has"),
            ("Doctor", "Department", "*", "1", "Association", "belongsTo"),
            ("MedicalRecord", "Prescription", "1", "*", "Association", "includes"),
            ("Doctor", "Prescription", "1", "*", "Association", "prescribes"),
        ],
        "enumerations": [
            {"name": "AppointmentStatus", "values": ["Scheduled", "Confirmed", "InProgress", "Completed", "Cancelled"]},
            {"name": "BloodType", "values": ["A_Pos", "A_Neg", "B_Pos", "B_Neg", "AB_Pos", "AB_Neg", "O_Pos", "O_Neg"]},
            {"name": "Gender", "values": ["Male", "Female", "Other"]},
        ],
        "design_advice": (
            "MedicalRecord is a composition owned by Patient (cannot exist independently). "
            "Keep Doctor-Patient relationship indirect via Appointment. "
            "Consider patient privacy: sensitive data should be in MedicalRecord, not Patient."
        ),
        "notes": (
            "Healthcare systems must consider patient privacy. Doctors belong to "
            "departments. Medical records are owned (composition) by patients. "
            "Appointments have statuses: Scheduled -> Confirmed -> Completed/Cancelled."
        ),
    },
    "university": {
        "keywords": [
            "university", "college", "school", "education", "student",
            "course", "enrollment", "academic", "campus",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Student", "attrs": ["id", "name", "email", "enrollmentDate", "gpa"]},
            {"name": "Professor", "attrs": ["id", "name", "email", "department", "title"]},
            {"name": "Course", "attrs": ["id", "name", "code", "credits", "description"]},
            {"name": "Enrollment", "attrs": ["enrollmentDate", "grade", "semester", "status"]},
            {"name": "Department", "attrs": ["id", "name", "building", "budget"]},
            {"name": "Grade", "attrs": ["id", "value", "letterGrade", "date"]},
            {"name": "Schedule", "attrs": ["dayOfWeek", "startTime", "endTime", "room"]},
        ],
        "key_relationships": [
            ("Student", "Enrollment", "1", "*", "Association", "registers"),
            ("Course", "Enrollment", "1", "*", "Association", "enrolledIn"),
            ("Professor", "Course", "1", "*", "Association", "teaches"),
            ("Course", "Department", "*", "1", "Association", "offeredBy"),
            ("Professor", "Department", "*", "1", "Association", "memberOf"),
            ("Course", "Schedule", "1", "*", "Composition", "scheduledAt"),
            ("Enrollment", "Grade", "1", "0..1", "Association", "receives"),
        ],
        "enumerations": [
            {"name": "EnrollmentStatus", "values": ["Enrolled", "Withdrawn", "Completed", "Failed"]},
            {"name": "Semester", "values": ["Fall", "Spring", "Summer"]},
            {"name": "AcademicTitle", "values": ["AssistantProfessor", "AssociateProfessor", "Professor", "Lecturer"]},
        ],
        "design_advice": (
            "Use Enrollment as an association class between Student and Course to avoid "
            "a direct many-to-many. Grade belongs to Enrollment. "
            "Consider Course prerequisites as a self-referencing relationship on Course."
        ),
        "notes": (
            "Use an Enrollment association class to avoid many-to-many between Student and Course. "
            "Enrollment holds grade and semester. Courses belong to departments. "
            "Consider prerequisites as a self-referencing relationship on Course."
        ),
    },
    "banking": {
        "keywords": [
            "bank", "banking", "finance", "financial", "account",
            "transaction", "atm", "loan",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Customer", "attrs": ["id", "name", "email", "phone", "address", "dateOfBirth"]},
            {"name": "Account", "attrs": ["accountNumber", "type", "balance", "openDate", "status"]},
            {"name": "Transaction", "attrs": ["id", "amount", "type", "timestamp", "description", "status"]},
            {"name": "Branch", "attrs": ["id", "name", "address", "phone"]},
            {"name": "Loan", "attrs": ["id", "amount", "interestRate", "startDate", "endDate", "status"]},
            {"name": "CreditCard", "attrs": ["cardNumber", "type", "expiryDate", "creditLimit", "status"]},
        ],
        "key_relationships": [
            ("Customer", "Account", "1", "1..*", "Association", "owns"),
            ("Account", "Transaction", "1", "*", "Composition", "records"),
            ("Customer", "Loan", "1", "*", "Association", "applies"),
            ("Account", "CreditCard", "1", "0..*", "Association", "linkedTo"),
            ("Branch", "Account", "1", "*", "Association", "manages"),
            ("Customer", "CreditCard", "1", "0..*", "Association", "holds"),
        ],
        "enumerations": [
            {"name": "AccountType", "values": ["Savings", "Checking", "Business", "Investment"]},
            {"name": "TransactionType", "values": ["Deposit", "Withdrawal", "Transfer", "Payment"]},
            {"name": "LoanStatus", "values": ["Pending", "Approved", "Active", "PaidOff", "Defaulted"]},
            {"name": "CardStatus", "values": ["Active", "Blocked", "Expired", "Cancelled"]},
        ],
        "design_advice": (
            "Consider Account as abstract with SavingsAccount and CheckingAccount subclasses. "
            "Transactions are compositions of Account (cannot exist without one). "
            "A customer must have at least one account (1..* multiplicity)."
        ),
        "notes": (
            "Banking: Account types include Savings, Checking, etc. "
            "Transaction types: Deposit, Withdrawal, Transfer. "
            "A customer must have at least one account. "
            "Consider Account as abstract with SavingsAccount and CheckingAccount subclasses."
        ),
    },
    "social_media": {
        "keywords": [
            "social media", "social network", "social platform",
            "twitter", "facebook", "instagram", "forum", "blog",
            "post", "feed", "follower",
        ],
        "synonyms": ["social", "network"],
        "core_classes": [
            {"name": "User", "attrs": ["id", "username", "email", "displayName", "bio", "joinDate"]},
            {"name": "Post", "attrs": ["id", "content", "timestamp", "likes", "visibility"]},
            {"name": "Comment", "attrs": ["id", "content", "timestamp"]},
            {"name": "Like", "attrs": ["id", "timestamp"]},
            {"name": "Follow", "attrs": ["id", "followDate"]},
            {"name": "Message", "attrs": ["id", "content", "timestamp", "isRead"]},
            {"name": "Notification", "attrs": ["id", "type", "content", "timestamp", "isRead"]},
        ],
        "key_relationships": [
            ("User", "Post", "1", "*", "Association", "publishes"),
            ("Post", "Comment", "1", "*", "Composition", "receives"),
            ("User", "Comment", "1", "*", "Association", "writes"),
            ("User", "Like", "1", "*", "Association", "gives"),
            ("Like", "Post", "*", "1", "Association", "on"),
            ("User", "Follow", "1", "*", "Association", "initiates"),
            ("Follow", "User", "*", "1", "Association", "targets"),
            ("User", "Message", "1", "*", "Association", "sends"),
            ("User", "Notification", "1", "*", "Association", "receives"),
        ],
        "enumerations": [
            {"name": "Visibility", "values": ["Public", "Private", "FriendsOnly"]},
            {"name": "NotificationType", "values": ["Like", "Comment", "Follow", "Mention", "Message"]},
        ],
        "design_advice": (
            "Model Follow as an explicit entity (not a direct User-User many-to-many) to "
            "track follow date and enable asymmetric following. Like should also be an entity "
            "to track who liked what and when. Comments are composed within Posts."
        ),
        "notes": (
            "Social media: User-follows-User is a self-referencing many-to-many. "
            "Posts can have media attachments. Comments are composed within posts. "
            "Like and Follow are modeled as entities for richer tracking."
        ),
    },
    "hotel": {
        "keywords": [
            "hotel", "reservation", "booking", "accommodation",
            "guest", "room", "hospitality",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Guest", "attrs": ["id", "name", "email", "phone", "idDocument"]},
            {"name": "Room", "attrs": ["roomNumber", "type", "floor", "pricePerNight", "status"]},
            {"name": "Reservation", "attrs": ["id", "checkInDate", "checkOutDate", "status", "totalCost"]},
            {"name": "Payment", "attrs": ["id", "amount", "method", "date", "status"]},
            {"name": "Service", "attrs": ["id", "name", "description", "price"]},
            {"name": "Staff", "attrs": ["id", "name", "role", "shift"]},
        ],
        "key_relationships": [
            ("Guest", "Reservation", "1", "*", "Association", "makes"),
            ("Reservation", "Room", "*", "1", "Association", "reserves"),
            ("Reservation", "Payment", "1", "1", "Association", "paidBy"),
            ("Reservation", "Service", "*", "*", "Association", "includes"),
            ("Staff", "Room", "1", "*", "Association", "manages"),
        ],
        "enumerations": [
            {"name": "RoomType", "values": ["Single", "Double", "Twin", "Suite", "Penthouse"]},
            {"name": "RoomStatus", "values": ["Available", "Occupied", "Maintenance", "Reserved"]},
            {"name": "ReservationStatus", "values": ["Pending", "Confirmed", "CheckedIn", "CheckedOut", "Cancelled"]},
        ],
        "design_advice": (
            "Room types should be an enumeration, not separate classes. "
            "Track room availability via RoomStatus. "
            "Services are optional add-ons (room service, spa, minibar)."
        ),
        "notes": (
            "Hotel: Room types include Single, Double, Suite, etc. "
            "Reservation status: Pending -> Confirmed -> CheckedIn -> CheckedOut. "
            "Services are optional add-ons (room service, spa, etc.)."
        ),
    },
    "restaurant": {
        "keywords": [
            "restaurant", "food", "menu", "dining", "chef",
            "waiter", "order food", "food delivery", "catering",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Customer", "attrs": ["id", "name", "phone", "email"]},
            {"name": "Menu", "attrs": ["id", "name", "description", "season"]},
            {"name": "MenuItem", "attrs": ["id", "name", "description", "price", "category", "isAvailable"]},
            {"name": "Order", "attrs": ["id", "orderDate", "status", "totalAmount", "tableNumber"]},
            {"name": "OrderItem", "attrs": ["quantity", "specialInstructions", "subtotal"]},
            {"name": "Table", "attrs": ["tableNumber", "capacity", "status", "location"]},
            {"name": "Chef", "attrs": ["id", "name", "speciality", "shift"]},
            {"name": "Reservation", "attrs": ["id", "dateTime", "partySize", "status"]},
        ],
        "key_relationships": [
            ("Customer", "Order", "1", "*", "Association", "places"),
            ("Order", "OrderItem", "1", "1..*", "Composition", "contains"),
            ("OrderItem", "MenuItem", "*", "1", "Association", "references"),
            ("Order", "Table", "*", "1", "Association", "assignedTo"),
            ("Customer", "Reservation", "1", "*", "Association", "makes"),
            ("Reservation", "Table", "*", "1", "Association", "reserves"),
            ("Menu", "MenuItem", "1", "*", "Composition", "includes"),
            ("Chef", "MenuItem", "1", "*", "Association", "prepares"),
        ],
        "enumerations": [
            {"name": "OrderStatus", "values": ["Received", "Preparing", "Ready", "Served", "Paid"]},
            {"name": "TableStatus", "values": ["Available", "Occupied", "Reserved", "OutOfService"]},
            {"name": "MenuCategory", "values": ["Appetizer", "MainCourse", "Dessert", "Beverage", "Side"]},
            {"name": "ReservationStatus", "values": ["Pending", "Confirmed", "Seated", "Cancelled", "NoShow"]},
        ],
        "design_advice": (
            "Separate MenuItem (on menu) from OrderItem (what was ordered). "
            "Menu can group MenuItems by season or time of day. "
            "Staff roles (Waiter, Chef, Manager) can be an enumeration or hierarchy."
        ),
        "notes": (
            "Restaurant: Separate MenuItem (on menu) from OrderItem (what was ordered). "
            "Tables have capacity and status (Available, Occupied, Reserved). "
            "Staff roles: Waiter, Chef, Manager, Host."
        ),
    },
    "inventory": {
        "keywords": [
            "inventory", "warehouse", "stock", "supply chain",
            "logistics", "shipment", "procurement",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Product", "attrs": ["id", "name", "sku", "description", "unitPrice", "weight"]},
            {"name": "Warehouse", "attrs": ["id", "name", "location", "capacity"]},
            {"name": "StockItem", "attrs": ["quantity", "reorderLevel", "lastRestocked"]},
            {"name": "Supplier", "attrs": ["id", "name", "contactEmail", "phone", "address"]},
            {"name": "PurchaseOrder", "attrs": ["id", "orderDate", "status", "totalAmount"]},
            {"name": "Shipment", "attrs": ["id", "shipDate", "estimatedArrival", "status", "trackingNumber"]},
        ],
        "key_relationships": [
            ("Warehouse", "StockItem", "1", "*", "Composition", "stores"),
            ("StockItem", "Product", "*", "1", "Association", "tracks"),
            ("Supplier", "Product", "1", "*", "Association", "supplies"),
            ("Supplier", "PurchaseOrder", "1", "*", "Association", "receives"),
            ("PurchaseOrder", "Shipment", "1", "0..1", "Association", "fulfilledBy"),
        ],
        "enumerations": [
            {"name": "OrderStatus", "values": ["Draft", "Submitted", "Approved", "Shipped", "Received"]},
            {"name": "ShipmentStatus", "values": ["Preparing", "InTransit", "Delivered", "Returned"]},
        ],
        "design_advice": (
            "StockItem is an association class connecting Product to Warehouse with quantity. "
            "Track reorder levels for automatic replenishment alerts. "
            "PurchaseOrder drives the procurement workflow."
        ),
        "notes": (
            "Inventory: StockItem connects Product to Warehouse with quantity. "
            "Track reorder levels for automatic replenishment. "
            "PurchaseOrder status: Draft -> Submitted -> Approved -> Shipped -> Received."
        ),
    },
    "project_management": {
        "keywords": [
            "project management", "project", "task", "kanban",
            "sprint", "agile", "scrum", "jira", "trello",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Project", "attrs": ["id", "name", "description", "startDate", "endDate", "status"]},
            {"name": "Task", "attrs": ["id", "title", "description", "priority", "status", "dueDate", "estimatedHours"]},
            {"name": "TeamMember", "attrs": ["id", "name", "email", "role"]},
            {"name": "Sprint", "attrs": ["id", "name", "startDate", "endDate", "goal"]},
            {"name": "Comment", "attrs": ["id", "content", "timestamp"]},
            {"name": "Team", "attrs": ["id", "name", "description"]},
        ],
        "key_relationships": [
            ("Project", "Task", "1", "*", "Composition", "contains"),
            ("Task", "TeamMember", "*", "1", "Association", "assignedTo"),
            ("Project", "Sprint", "1", "*", "Composition", "organizedIn"),
            ("Sprint", "Task", "1", "*", "Association", "includes"),
            ("Task", "Comment", "1", "*", "Composition", "has"),
            ("Team", "TeamMember", "1", "*", "Association", "comprises"),
            ("Team", "Project", "1", "*", "Association", "worksOn"),
        ],
        "enumerations": [
            {"name": "TaskStatus", "values": ["Todo", "InProgress", "Review", "Done", "Blocked"]},
            {"name": "Priority", "values": ["Low", "Medium", "High", "Critical"]},
            {"name": "ProjectStatus", "values": ["Planning", "Active", "OnHold", "Completed", "Cancelled"]},
        ],
        "design_advice": (
            "Tasks are composed within Projects. Sprints group tasks into time-boxed iterations. "
            "Use Priority and TaskStatus enumerations instead of free-text strings. "
            "Consider task dependencies as a self-referencing relationship."
        ),
        "notes": (
            "Project management: Tasks have statuses (Todo, InProgress, Review, Done). "
            "Priority levels: Low, Medium, High, Critical. "
            "Sprints are time-boxed iterations containing tasks."
        ),
    },
    "hr": {
        "keywords": [
            "hr", "human resource", "human resources", "employee",
            "payroll", "salary", "leave", "recruitment", "hiring",
            "workforce",
        ],
        "synonyms": [],
        "core_classes": [
            {"name": "Employee", "attrs": ["id", "name", "email", "phone", "hireDate", "status"]},
            {"name": "Department", "attrs": ["id", "name", "location", "budget"]},
            {"name": "Position", "attrs": ["id", "title", "description", "level", "salaryRange"]},
            {"name": "Salary", "attrs": ["id", "amount", "currency", "effectiveDate", "type"]},
            {"name": "LeaveRequest", "attrs": ["id", "startDate", "endDate", "type", "status", "reason"]},
            {"name": "Performance", "attrs": ["id", "reviewDate", "rating", "goals", "feedback"]},
        ],
        "key_relationships": [
            ("Employee", "Department", "*", "1", "Association", "belongsTo"),
            ("Employee", "Position", "*", "1", "Association", "holds"),
            ("Employee", "Salary", "1", "*", "Composition", "earns"),
            ("Employee", "LeaveRequest", "1", "*", "Association", "submits"),
            ("Employee", "Performance", "1", "*", "Composition", "receives"),
            ("Department", "Employee", "1", "0..1", "Association", "managedBy"),
        ],
        "enumerations": [
            {"name": "EmployeeStatus", "values": ["Active", "OnLeave", "Terminated", "Retired"]},
            {"name": "LeaveType", "values": ["Annual", "Sick", "Maternity", "Paternity", "Unpaid"]},
            {"name": "LeaveStatus", "values": ["Pending", "Approved", "Rejected", "Cancelled"]},
            {"name": "PerformanceRating", "values": ["Exceptional", "ExceedsExpectations", "MeetsExpectations", "NeedsImprovement"]},
        ],
        "design_advice": (
            "Salary is a composition of Employee to track history (effective dates). "
            "Performance reviews should reference goals set in prior reviews. "
            "Department has a managedBy relationship back to Employee (the manager)."
        ),
        "notes": (
            "HR systems: Track salary history with effective dates, not just current salary. "
            "Leave requests have approval workflows. Performance reviews are periodic. "
            "Department manager is an Employee (self-referencing pattern)."
        ),
    },
}

# ---------------------------------------------------------------------------
# Synonym expansion map (built from patterns at import time)
# ---------------------------------------------------------------------------

_SYNONYM_TO_PATTERN: Dict[str, str] = {}
for _pname, _pdata in DOMAIN_PATTERNS.items():
    for _syn in _pdata.get("synonyms", []):
        _SYNONYM_TO_PATTERN[_syn.lower()] = _pname


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def detect_domain_pattern(user_message: str) -> Optional[Dict[str, Any]]:
    """Detect a matching domain pattern from the user's message.

    Returns the pattern dict if a match is found, ``None`` otherwise.
    Matches on keyword presence with word-boundary awareness for short words,
    and also checks synonyms.
    """
    if not isinstance(user_message, str):
        return None

    message_lower = user_message.lower()
    best_match: Optional[str] = None
    best_score = 0

    for pattern_name, pattern_data in DOMAIN_PATTERNS.items():
        score = 0
        for keyword in pattern_data["keywords"]:
            if len(keyword) <= 4:
                # Use word boundary for short keywords to avoid false positives
                if re.search(rf'\b{re.escape(keyword)}\b', message_lower):
                    score += 1
            else:
                if keyword in message_lower:
                    score += 1

        # Check synonyms with word-boundary matching
        for synonym in pattern_data.get("synonyms", []):
            if re.search(rf'\b{re.escape(synonym.lower())}\b', message_lower):
                score += 1

        if score > best_score:
            best_score = score
            best_match = pattern_name

    if best_match and best_score >= 1:
        logger.info(f"[DomainPatterns] Matched pattern '{best_match}' (score={best_score})")
        return DOMAIN_PATTERNS[best_match]

    return None


def format_pattern_for_prompt(pattern: Dict[str, Any]) -> str:
    """Format a domain pattern as a reference block for the LLM prompt.

    This gives the LLM a strong hint about what classes and relationships
    are typically expected for this domain, dramatically improving output quality.
    """
    lines: List[str] = []
    lines.append("DOMAIN REFERENCE (use as inspiration, adapt to user's specific request):")

    # Core classes with attributes
    lines.append("Typical classes for this domain:")
    for cls in pattern.get("core_classes", []):
        attrs = ", ".join(cls["attrs"][:8])
        lines.append(f"  - {cls['name']}: {attrs}")

    # Key relationships
    lines.append("Typical relationships:")
    for rel in pattern.get("key_relationships", []):
        source, target, src_mult, tgt_mult, rel_type, name = rel
        lines.append(
            f"  - {source} -> {target} ({rel_type}, {src_mult}..{tgt_mult}, \"{name}\")"
        )

    # Enumerations
    enums = pattern.get("enumerations", [])
    if enums:
        lines.append("Common enumerations:")
        for enum in enums:
            values = ", ".join(enum["values"][:6])
            lines.append(f"  - {enum['name']}: {values}")

    # Design advice
    advice = pattern.get("design_advice")
    if advice:
        lines.append(f"Design advice: {advice}")

    # Notes
    notes = pattern.get("notes")
    if notes:
        lines.append(f"Domain notes: {notes}")

    lines.append(
        "IMPORTANT: This is a reference only. Follow the user's request strictly. "
        "Add or remove classes based on what they ask for. Use the reference to ensure "
        "you don't miss critical relationships and that multiplicities are correct."
    )

    return "\n".join(lines)


def get_pattern_hint(user_message: str) -> str:
    """Return a formatted pattern hint for the user's message, or empty string."""
    pattern = detect_domain_pattern(user_message)
    if pattern is None:
        return ""
    return "\n\n" + format_pattern_for_prompt(pattern)
