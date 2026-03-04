"""
State Machine Behavioral Patterns
----------------------------------
Common behavioral patterns for state machine diagram generation.
When a user request matches a known domain, the matching pattern is
injected into the LLM prompt so transitions, guards, and effects are
domain-accurate from the first pass.

Mirrors the approach of ``domain_patterns.py`` for ClassDiagrams.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

STATE_MACHINE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "order_processing": {
        "keywords": [
            "order", "purchase", "checkout", "e-commerce", "ecommerce",
            "shopping", "buy", "cart", "fulfillment",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "PendingPayment", "type": "regular", "entry": "display payment options", "do": "await payment"},
            {"name": "PaymentProcessing", "type": "regular", "entry": "charge payment method", "do": "validate transaction"},
            {"name": "Confirmed", "type": "regular", "entry": "send confirmation email", "do": "update inventory"},
            {"name": "Preparing", "type": "regular", "entry": "assign to warehouse", "do": "pick and pack items"},
            {"name": "Shipped", "type": "regular", "entry": "generate tracking number", "do": "track shipment"},
            {"name": "Delivered", "type": "regular", "entry": "send delivery notification"},
            {"name": "Cancelled", "type": "regular", "entry": "process refund"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "PendingPayment", "trigger": "placeOrder"},
            {"source": "PendingPayment", "target": "PaymentProcessing", "trigger": "submitPayment"},
            {"source": "PendingPayment", "target": "Cancelled", "trigger": "cancelOrder"},
            {"source": "PaymentProcessing", "target": "Confirmed", "trigger": "paymentSucceeded"},
            {"source": "PaymentProcessing", "target": "PendingPayment", "trigger": "paymentFailed", "effect": "notify customer"},
            {"source": "Confirmed", "target": "Preparing", "trigger": "startFulfillment"},
            {"source": "Confirmed", "target": "Cancelled", "trigger": "cancelOrder", "guard": "not yet shipped"},
            {"source": "Preparing", "target": "Shipped", "trigger": "shipOrder"},
            {"source": "Shipped", "target": "Delivered", "trigger": "confirmDelivery"},
            {"source": "Delivered", "target": "Final", "trigger": "complete"},
            {"source": "Cancelled", "target": "Final", "trigger": "complete"},
        ],
        "notes": "Order processing needs clear payment states and a cancellation path from pre-shipment states.",
    },
    "authentication": {
        "keywords": [
            "auth", "login", "authentication", "session", "sign in",
            "sign up", "register", "password", "credential", "oauth",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "Unauthenticated", "type": "regular", "do": "display login form"},
            {"name": "ValidatingCredentials", "type": "regular", "entry": "hash password", "do": "check credentials"},
            {"name": "MFAChallenge", "type": "regular", "entry": "send verification code", "do": "await code input"},
            {"name": "Authenticated", "type": "regular", "entry": "create session token", "do": "monitor session"},
            {"name": "SessionExpired", "type": "regular", "entry": "clear session"},
            {"name": "Locked", "type": "regular", "entry": "lock account", "do": "start cooldown timer"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "Unauthenticated", "trigger": "start"},
            {"source": "Unauthenticated", "target": "ValidatingCredentials", "trigger": "submitCredentials"},
            {"source": "ValidatingCredentials", "target": "Authenticated", "trigger": "credentialsValid", "guard": "MFA not required"},
            {"source": "ValidatingCredentials", "target": "MFAChallenge", "trigger": "credentialsValid", "guard": "MFA required"},
            {"source": "ValidatingCredentials", "target": "Unauthenticated", "trigger": "credentialsInvalid", "effect": "increment attempt counter"},
            {"source": "ValidatingCredentials", "target": "Locked", "trigger": "credentialsInvalid", "guard": "attempts >= max"},
            {"source": "MFAChallenge", "target": "Authenticated", "trigger": "codeVerified"},
            {"source": "MFAChallenge", "target": "Unauthenticated", "trigger": "codeFailed"},
            {"source": "Authenticated", "target": "SessionExpired", "trigger": "sessionTimeout"},
            {"source": "Authenticated", "target": "Unauthenticated", "trigger": "logout", "effect": "invalidate session"},
            {"source": "SessionExpired", "target": "Unauthenticated", "trigger": "redirectToLogin"},
            {"source": "Locked", "target": "Unauthenticated", "trigger": "cooldownExpired"},
            {"source": "Unauthenticated", "target": "Final", "trigger": "exit"},
        ],
        "notes": "Authentication flows need MFA branching, account lockout after failed attempts, and session lifecycle management.",
    },
    "document_workflow": {
        "keywords": [
            "document", "approval", "review", "workflow", "publish",
            "draft", "article", "content", "editorial", "manuscript",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "Draft", "type": "regular", "do": "allow editing"},
            {"name": "UnderReview", "type": "regular", "entry": "notify reviewers", "do": "await review"},
            {"name": "RevisionRequested", "type": "regular", "entry": "send feedback to author"},
            {"name": "Approved", "type": "regular", "entry": "mark as approved"},
            {"name": "Published", "type": "regular", "entry": "make publicly visible"},
            {"name": "Archived", "type": "regular", "entry": "move to archive"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "Draft", "trigger": "createDocument"},
            {"source": "Draft", "target": "UnderReview", "trigger": "submitForReview"},
            {"source": "UnderReview", "target": "Approved", "trigger": "approve"},
            {"source": "UnderReview", "target": "RevisionRequested", "trigger": "requestRevision", "effect": "add review comments"},
            {"source": "RevisionRequested", "target": "Draft", "trigger": "startRevision"},
            {"source": "Approved", "target": "Published", "trigger": "publish"},
            {"source": "Published", "target": "Archived", "trigger": "archive"},
            {"source": "Published", "target": "Draft", "trigger": "unpublish", "effect": "create new version"},
            {"source": "Archived", "target": "Final", "trigger": "delete"},
        ],
        "notes": "Document workflows need revision loops and clear approval gates.",
    },
    "task_management": {
        "keywords": [
            "task", "issue", "ticket", "backlog", "sprint",
            "kanban", "todo", "assignment", "project management",
            "bug", "feature request", "jira", "trello",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "Backlog", "type": "regular", "entry": "add to backlog"},
            {"name": "Todo", "type": "regular", "entry": "prioritize task"},
            {"name": "InProgress", "type": "regular", "entry": "assign developer", "do": "track progress"},
            {"name": "InReview", "type": "regular", "entry": "create pull request", "do": "await code review"},
            {"name": "Testing", "type": "regular", "entry": "assign QA", "do": "run test suite"},
            {"name": "Done", "type": "regular", "entry": "close ticket"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "Backlog", "trigger": "createTask"},
            {"source": "Backlog", "target": "Todo", "trigger": "planSprint"},
            {"source": "Todo", "target": "InProgress", "trigger": "startWork"},
            {"source": "InProgress", "target": "InReview", "trigger": "submitForReview"},
            {"source": "InReview", "target": "InProgress", "trigger": "changesRequested"},
            {"source": "InReview", "target": "Testing", "trigger": "approved"},
            {"source": "Testing", "target": "InProgress", "trigger": "bugFound", "effect": "reopen task"},
            {"source": "Testing", "target": "Done", "trigger": "testsPassed"},
            {"source": "Done", "target": "Final", "trigger": "archived"},
        ],
        "notes": "Task management flows need review/testing feedback loops and clear handoff triggers.",
    },
    "booking_reservation": {
        "keywords": [
            "booking", "reservation", "hotel", "flight", "appointment",
            "schedule", "room", "book", "reserve", "availability",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "Browsing", "type": "regular", "do": "display available options"},
            {"name": "Selected", "type": "regular", "entry": "hold reservation temporarily"},
            {"name": "PaymentPending", "type": "regular", "do": "await payment"},
            {"name": "Confirmed", "type": "regular", "entry": "send confirmation", "do": "block calendar slot"},
            {"name": "CheckedIn", "type": "regular", "entry": "mark as active"},
            {"name": "Completed", "type": "regular", "entry": "request feedback"},
            {"name": "Cancelled", "type": "regular", "entry": "process refund", "effect": "release slot"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "Browsing", "trigger": "searchAvailability"},
            {"source": "Browsing", "target": "Selected", "trigger": "selectOption"},
            {"source": "Selected", "target": "PaymentPending", "trigger": "proceedToPayment"},
            {"source": "Selected", "target": "Browsing", "trigger": "holdExpired"},
            {"source": "PaymentPending", "target": "Confirmed", "trigger": "paymentSucceeded"},
            {"source": "PaymentPending", "target": "Browsing", "trigger": "paymentFailed"},
            {"source": "Confirmed", "target": "CheckedIn", "trigger": "checkIn"},
            {"source": "Confirmed", "target": "Cancelled", "trigger": "cancelBooking", "guard": "within cancellation window"},
            {"source": "CheckedIn", "target": "Completed", "trigger": "checkOut"},
            {"source": "Completed", "target": "Final", "trigger": "close"},
            {"source": "Cancelled", "target": "Final", "trigger": "close"},
        ],
        "notes": "Booking flows need temporary holds, cancellation windows, and check-in/check-out lifecycle.",
    },
    "user_registration": {
        "keywords": [
            "registration", "onboarding", "sign up", "account creation",
            "profile setup", "verification", "activation",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "FormEntry", "type": "regular", "do": "collect user information"},
            {"name": "EmailVerification", "type": "regular", "entry": "send verification email", "do": "await verification"},
            {"name": "ProfileSetup", "type": "regular", "do": "collect additional profile data"},
            {"name": "Active", "type": "regular", "entry": "activate account"},
            {"name": "Suspended", "type": "regular", "entry": "restrict access"},
            {"name": "Deactivated", "type": "regular", "entry": "anonymize data"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "FormEntry", "trigger": "startRegistration"},
            {"source": "FormEntry", "target": "EmailVerification", "trigger": "submitForm", "guard": "form valid"},
            {"source": "EmailVerification", "target": "ProfileSetup", "trigger": "emailVerified"},
            {"source": "EmailVerification", "target": "FormEntry", "trigger": "verificationExpired"},
            {"source": "ProfileSetup", "target": "Active", "trigger": "completeProfile"},
            {"source": "Active", "target": "Suspended", "trigger": "violatePolicy"},
            {"source": "Suspended", "target": "Active", "trigger": "appealAccepted"},
            {"source": "Active", "target": "Deactivated", "trigger": "deleteAccount"},
            {"source": "Suspended", "target": "Deactivated", "trigger": "permanentBan"},
            {"source": "Deactivated", "target": "Final", "trigger": "dataDeleted"},
        ],
        "notes": "Registration flows need email verification, profile completion, and account lifecycle states.",
    },
    "payment_processing": {
        "keywords": [
            "payment", "transaction", "billing", "invoice",
            "charge", "refund", "stripe", "paypal",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "Initiated", "type": "regular", "entry": "create transaction record"},
            {"name": "Authorizing", "type": "regular", "entry": "call payment gateway", "do": "await authorization"},
            {"name": "Authorized", "type": "regular", "entry": "hold funds"},
            {"name": "Capturing", "type": "regular", "entry": "capture authorized amount"},
            {"name": "Completed", "type": "regular", "entry": "update ledger"},
            {"name": "Failed", "type": "regular", "entry": "log failure reason"},
            {"name": "Refunding", "type": "regular", "entry": "initiate refund"},
            {"name": "Refunded", "type": "regular", "entry": "credit customer account"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "Initiated", "trigger": "createPayment"},
            {"source": "Initiated", "target": "Authorizing", "trigger": "authorize"},
            {"source": "Authorizing", "target": "Authorized", "trigger": "authorizationSucceeded"},
            {"source": "Authorizing", "target": "Failed", "trigger": "authorizationFailed"},
            {"source": "Authorized", "target": "Capturing", "trigger": "capturePayment"},
            {"source": "Authorized", "target": "Failed", "trigger": "authorizationExpired"},
            {"source": "Capturing", "target": "Completed", "trigger": "captureSucceeded"},
            {"source": "Capturing", "target": "Failed", "trigger": "captureFailed"},
            {"source": "Completed", "target": "Refunding", "trigger": "requestRefund"},
            {"source": "Refunding", "target": "Refunded", "trigger": "refundProcessed"},
            {"source": "Failed", "target": "Initiated", "trigger": "retry"},
            {"source": "Refunded", "target": "Final", "trigger": "close"},
            {"source": "Failed", "target": "Final", "trigger": "abandon"},
        ],
        "notes": "Payment flows need authorization-capture separation, refund paths, and failure recovery.",
    },
    "support_ticket": {
        "keywords": [
            "support", "helpdesk", "customer service", "ticket",
            "incident", "escalation", "sla", "help desk",
        ],
        "states": [
            {"name": "Initial", "type": "initial"},
            {"name": "New", "type": "regular", "entry": "assign ticket number"},
            {"name": "Triaged", "type": "regular", "entry": "assign priority and category"},
            {"name": "Assigned", "type": "regular", "entry": "notify assigned agent"},
            {"name": "InProgress", "type": "regular", "do": "work on resolution"},
            {"name": "WaitingOnCustomer", "type": "regular", "entry": "send follow-up question"},
            {"name": "Escalated", "type": "regular", "entry": "notify senior agent"},
            {"name": "Resolved", "type": "regular", "entry": "send resolution to customer"},
            {"name": "Closed", "type": "regular", "entry": "archive ticket"},
            {"name": "Final", "type": "final"},
        ],
        "transitions": [
            {"source": "Initial", "target": "New", "trigger": "submitTicket"},
            {"source": "New", "target": "Triaged", "trigger": "triage"},
            {"source": "Triaged", "target": "Assigned", "trigger": "assignAgent"},
            {"source": "Assigned", "target": "InProgress", "trigger": "startWork"},
            {"source": "InProgress", "target": "WaitingOnCustomer", "trigger": "needMoreInfo"},
            {"source": "WaitingOnCustomer", "target": "InProgress", "trigger": "customerReplied"},
            {"source": "InProgress", "target": "Escalated", "trigger": "escalate", "guard": "SLA breached or complex issue"},
            {"source": "Escalated", "target": "InProgress", "trigger": "takeOver"},
            {"source": "InProgress", "target": "Resolved", "trigger": "resolve"},
            {"source": "Resolved", "target": "Closed", "trigger": "customerConfirmed"},
            {"source": "Resolved", "target": "InProgress", "trigger": "reopen", "guard": "within reopen window"},
            {"source": "Closed", "target": "Final", "trigger": "archive"},
        ],
        "notes": "Support ticket flows need escalation paths, customer-wait states, and SLA awareness.",
    },
}


# ---------------------------------------------------------------------------
# Detection & formatting helpers
# ---------------------------------------------------------------------------

def detect_state_pattern(user_request: str) -> Optional[str]:
    """Detect if a user request matches a known state machine behavioral pattern.

    Returns the pattern key (e.g. ``"order_processing"``) or ``None``.
    """
    request_lower = user_request.lower()
    best_match: Optional[str] = None
    best_score = 0

    for pattern_key, pattern in STATE_MACHINE_PATTERNS.items():
        score = sum(1 for kw in pattern["keywords"] if kw in request_lower)
        if score > best_score:
            best_score = score
            best_match = pattern_key

    if best_score >= 1:
        logger.info(f"[StatePatterns] Detected pattern '{best_match}' (score={best_score})")
        return best_match
    return None


def format_state_pattern_for_prompt(pattern_key: str) -> str:
    """Format a state machine pattern as an LLM prompt reference block."""
    pattern = STATE_MACHINE_PATTERNS.get(pattern_key)
    if not pattern:
        return ""

    lines: List[str] = [
        f"\n\nDomain Reference ({pattern_key.replace('_', ' ').title()}):",
        "Use this as a guide — adapt to the user's specific request but keep these"
        " domain-accurate states and transitions as a baseline.\n",
    ]

    lines.append("Reference states:")
    for s in pattern["states"]:
        parts = [f"  - {s['name']} (type: {s['type']})"]
        if s.get("entry"):
            parts.append(f" | entry: {s['entry']}")
        if s.get("do"):
            parts.append(f" | do: {s['do']}")
        lines.append("".join(parts))

    lines.append("\nReference transitions:")
    for t in pattern["transitions"]:
        desc = f"  - {t['source']} -> {t['target']} [trigger: {t['trigger']}]"
        if t.get("guard"):
            desc += f" [guard: {t['guard']}]"
        if t.get("effect"):
            desc += f" [effect: {t['effect']}]"
        lines.append(desc)

    if pattern.get("notes"):
        lines.append(f"\nDesign note: {pattern['notes']}")

    return "\n".join(lines)


def get_state_pattern_hint(user_request: str) -> str:
    """Convenience function: detect pattern and return formatted hint.

    Returns an empty string if no pattern matches.
    """
    pattern_key = detect_state_pattern(user_request)
    if pattern_key:
        return format_state_pattern_for_prompt(pattern_key)
    return ""
