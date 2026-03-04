"""
Domain Pattern Library
----------------------
Common domain patterns for well-known system types.  When the user's request
matches a known domain, the relevant pattern is injected into the LLM prompt
to dramatically improve the quality of generated class diagrams.

Each pattern contains:
- ``keywords``: terms that trigger this pattern (checked against user message)
- ``core_classes``: the essential classes with typical attributes
- ``key_relationships``: common relationships between classes
- ``notes``: domain-specific modeling tips for the LLM
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
        "core_classes": [
            {"name": "Student", "attrs": ["id", "name", "email", "enrollmentDate", "gpa"]},
            {"name": "Professor", "attrs": ["id", "name", "email", "department", "title"]},
            {"name": "Course", "attrs": ["id", "name", "code", "credits", "description"]},
            {"name": "Enrollment", "attrs": ["enrollmentDate", "grade", "semester", "status"]},
            {"name": "Department", "attrs": ["id", "name", "building", "budget"]},
            {"name": "Schedule", "attrs": ["dayOfWeek", "startTime", "endTime", "room"]},
        ],
        "key_relationships": [
            ("Student", "Enrollment", "1", "*", "Association", "registers"),
            ("Course", "Enrollment", "1", "*", "Association", "enrolledIn"),
            ("Professor", "Course", "1", "*", "Association", "teaches"),
            ("Course", "Department", "*", "1", "Association", "offeredBy"),
            ("Professor", "Department", "*", "1", "Association", "memberOf"),
            ("Course", "Schedule", "1", "*", "Composition", "scheduledAt"),
        ],
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
        "core_classes": [
            {"name": "Customer", "attrs": ["id", "name", "email", "phone", "address", "dateOfBirth"]},
            {"name": "Account", "attrs": ["accountNumber", "type", "balance", "openDate", "status"]},
            {"name": "Transaction", "attrs": ["id", "amount", "type", "timestamp", "description", "status"]},
            {"name": "Branch", "attrs": ["id", "name", "address", "phone"]},
            {"name": "Loan", "attrs": ["id", "amount", "interestRate", "startDate", "endDate", "status"]},
            {"name": "Card", "attrs": ["cardNumber", "type", "expiryDate", "status"]},
        ],
        "key_relationships": [
            ("Customer", "Account", "1", "1..*", "Association", "owns"),
            ("Account", "Transaction", "1", "*", "Composition", "records"),
            ("Customer", "Loan", "1", "*", "Association", "applies"),
            ("Account", "Card", "1", "0..*", "Association", "linkedTo"),
            ("Branch", "Account", "1", "*", "Association", "manages"),
        ],
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
        "core_classes": [
            {"name": "User", "attrs": ["id", "username", "email", "displayName", "bio", "joinDate"]},
            {"name": "Post", "attrs": ["id", "content", "timestamp", "likes", "visibility"]},
            {"name": "Comment", "attrs": ["id", "content", "timestamp"]},
            {"name": "Message", "attrs": ["id", "content", "timestamp", "isRead"]},
            {"name": "Group", "attrs": ["id", "name", "description", "createdDate"]},
            {"name": "Notification", "attrs": ["id", "type", "content", "timestamp", "isRead"]},
        ],
        "key_relationships": [
            ("User", "Post", "1", "*", "Association", "publishes"),
            ("Post", "Comment", "1", "*", "Composition", "receives"),
            ("User", "Comment", "1", "*", "Association", "writes"),
            ("User", "User", "*", "*", "Association", "follows"),
            ("User", "Message", "1", "*", "Association", "sends"),
            ("User", "Group", "*", "*", "Association", "memberOf"),
            ("User", "Notification", "1", "*", "Association", "receives"),
        ],
        "notes": (
            "Social media: User-follows-User is a self-referencing many-to-many. "
            "Posts can have media attachments. Comments are composed within posts. "
            "Consider a Like entity for tracking who liked what."
        ),
    },
    "hotel": {
        "keywords": [
            "hotel", "reservation", "booking", "accommodation",
            "guest", "room", "hospitality",
        ],
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
        "core_classes": [
            {"name": "Customer", "attrs": ["id", "name", "phone", "email"]},
            {"name": "MenuItem", "attrs": ["id", "name", "description", "price", "category", "isAvailable"]},
            {"name": "Order", "attrs": ["id", "orderDate", "status", "totalAmount", "tableNumber"]},
            {"name": "OrderItem", "attrs": ["quantity", "specialInstructions", "subtotal"]},
            {"name": "Table", "attrs": ["tableNumber", "capacity", "status", "location"]},
            {"name": "Staff", "attrs": ["id", "name", "role", "shift"]},
            {"name": "Reservation", "attrs": ["id", "dateTime", "partySize", "status"]},
        ],
        "key_relationships": [
            ("Customer", "Order", "1", "*", "Association", "places"),
            ("Order", "OrderItem", "1", "1..*", "Composition", "contains"),
            ("OrderItem", "MenuItem", "*", "1", "Association", "references"),
            ("Order", "Table", "*", "1", "Association", "assignedTo"),
            ("Customer", "Reservation", "1", "*", "Association", "makes"),
            ("Reservation", "Table", "*", "1", "Association", "reserves"),
        ],
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
        "notes": (
            "Project management: Tasks have statuses (Todo, InProgress, Review, Done). "
            "Priority levels: Low, Medium, High, Critical. "
            "Sprints are time-boxed iterations containing tasks."
        ),
    },
}


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def detect_domain_pattern(user_message: str) -> Optional[Dict[str, Any]]:
    """Detect a matching domain pattern from the user's message.

    Returns the pattern dict if a match is found, ``None`` otherwise.
    Matches on keyword presence with word-boundary awareness for short words.
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
                # Use word boundary for short keywords
                if re.search(rf'\b{re.escape(keyword)}\b', message_lower):
                    score += 1
            else:
                if keyword in message_lower:
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

    # Core classes
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
