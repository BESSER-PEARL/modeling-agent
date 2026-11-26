"""Structured-output schemas for the requirements stated in an application request.

The kinds are BESSER's ``RequirementKind`` values
(``besser/BUML/metamodel/traceability/traceability.py``); the editor records
each requirement as a ``RequirementArtifact`` in the project trace.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

RequirementKind = Literal[
    "data", "uniqueness", "validation", "rule", "computed", "transition",
    "action", "ui", "integration", "quality", "technical",
]


class ExtractedRequirementSpec(BaseModel):
    """One atomic requirement the request states."""
    text: str = Field(description="The requirement as ONE atomic statement, in the user's words where possible.")
    quote: str = Field(
        description="The words of the request that state it, copied EXACTLY (a phrase or a sentence, "
                    "never a paragraph and never reworded).",
    )
    kind: RequirementKind = Field(
        description="data: an entity with the attributes stated for it, or a relationship between entities; "
                    "uniqueness: an identifier or a value that must be unique; validation: the shape a value "
                    "must have; rule: a limit or a rule across entities; computed: a value worked out rather "
                    "than typed in; transition: a state change and when it is refused; action: a named "
                    "operation; ui: a screen or interaction; integration: an external service; quality: "
                    "performance, security, accessibility or another quality of the whole application; "
                    "technical: a technology, platform or architecture to use.",
    )


class RequirementsExtractionSpec(BaseModel):
    """Every requirement the request states, in the order it states them."""
    requirements: List[ExtractedRequirementSpec] = Field(
        default_factory=list,
        description="The requirements; EMPTY when the request states none.",
    )


class RealizationSpec(BaseModel):
    """The elements of the class diagram that realize one requirement."""
    requirement: str = Field(description="The requirement's id from the list, e.g. 'R2'.")
    elements: List[str] = Field(
        description="Each element as 'Class' (a class or enumeration), 'Class.member' (an attribute, an "
                    "enumeration literal or an association end), 'Class.method()' (a method) or "
                    "'inv name' (an OCL invariant).",
    )
