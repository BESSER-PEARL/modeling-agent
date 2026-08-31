"""
Pydantic schemas for OpenAI structured outputs.

Each diagram type has dedicated request/response schemas that guarantee
valid JSON from the LLM — eliminating the need for manual JSON parsing,
repair loops, and retry-with-suffix hacks.
"""

from .class_diagram import (
    AttributeSpec,
    MethodParameterSpec,
    MethodSpec,
    RelationshipSpec,
    OCLConstraintSpec,
    SingleClassSpec,
    SystemClassSpec,
    ClassModificationTarget,
    ClassModificationChanges,
    ClassModification,
    ClassModificationResponse,
)
from .state_machine import (
    StateSpec,
    TransitionSpec,
    SingleStateSpec,
    SystemStateMachineSpec,
    StateMachineModificationTarget,
    StateMachineModificationChanges,
    StateMachineModification,
    StateMachineModificationResponse,
)
from .object_diagram import (
    ObjectAttributeSpec,
    SingleObjectSpec,
    ObjectLinkSpec,
    SystemObjectSpec,
    ObjectModificationTarget,
    ObjectModificationChanges,
    ObjectModification,
    ObjectModificationResponse,
)
from .agent_diagram import (
    AgentReplySpec,
    AgentStateSpec,
    AgentIntentSpec,
    AgentSingleElementSpec,
    AgentTransitionSpec,
    SystemAgentSpec,
    AgentModificationTarget,
    AgentModificationChanges,
    AgentModification,
    AgentModificationResponse,
)
from .gui_diagram import (
    GUISectionSpec,
    GUIBindSpec,
    SingleGUIElementSpec,
    GUIPageSpec,
    SystemGUISpec,
    AuthoredGUISectionSpec,
    AuthoredGUIPageSpec,
    AuthoredSystemGUISpec,
    GUIThemeSpec,
    GUIModificationSpec,
    GUIModificationBatchSpec,
)
from .quantum_circuit import (
    QuantumOperationSpec,
    SingleQuantumGateSpec,
    SystemQuantumCircuitSpec,
    QuantumModificationSpec,
)
from .bpmn import (
    BPMNNodeSpec,
    BPMNFlowSpec,
    BPMNLaneSpec,
    BPMNPoolSpec,
    SystemBPMNSpec,
    BPMNModificationTarget,
    BPMNModificationChanges,
    BPMNModification,
    BPMNModificationResponse,
)

__all__ = [
    # Class Diagram
    "AttributeSpec", "MethodParameterSpec", "MethodSpec",
    "RelationshipSpec", "OCLConstraintSpec", "SingleClassSpec", "SystemClassSpec",
    "ClassModificationTarget", "ClassModificationChanges",
    "ClassModification", "ClassModificationResponse",
    # State Machine
    "StateSpec", "TransitionSpec", "SingleStateSpec", "SystemStateMachineSpec",
    "StateMachineModificationTarget", "StateMachineModificationChanges",
    "StateMachineModification", "StateMachineModificationResponse",
    # Object Diagram
    "ObjectAttributeSpec", "SingleObjectSpec", "ObjectLinkSpec", "SystemObjectSpec",
    "ObjectModificationTarget", "ObjectModificationChanges",
    "ObjectModification", "ObjectModificationResponse",
    # Agent Diagram
    "AgentReplySpec", "AgentStateSpec", "AgentIntentSpec",
    "AgentSingleElementSpec", "AgentTransitionSpec", "SystemAgentSpec",
    "AgentModificationTarget", "AgentModificationChanges",
    "AgentModification", "AgentModificationResponse",
    # GUI Diagram
    "GUISectionSpec", "GUIBindSpec", "SingleGUIElementSpec", "GUIPageSpec",
    "AuthoredGUISectionSpec", "AuthoredGUIPageSpec", "AuthoredSystemGUISpec",
    "GUIThemeSpec",
    "GUIModificationBatchSpec",
    "SystemGUISpec", "GUIModificationSpec",
    # Quantum Circuit
    "QuantumOperationSpec", "SingleQuantumGateSpec", "SystemQuantumCircuitSpec",
    "QuantumModificationSpec",
    # BPMN
    "BPMNNodeSpec", "BPMNFlowSpec", "BPMNLaneSpec", "BPMNPoolSpec", "SystemBPMNSpec",
    "BPMNModificationTarget", "BPMNModificationChanges",
    "BPMNModification", "BPMNModificationResponse",
]
