"""Pydantic schemas for Class Diagram structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import re

from pydantic import BaseModel, Field, field_validator, model_validator


class MethodParameterSpec(BaseModel):
    name: str = Field(min_length=1, max_length=50, description="Parameter name in camelCase")
    type: str = Field(default="String", description="Parameter type: String, int, boolean, float, Date, or a custom class name")


class AttributeSpec(BaseModel):
    name: str = Field(min_length=1, max_length=50, description="Attribute name in camelCase")
    type: Optional[str] = Field(default=None, description="Data type (e.g. String, int, bool, float, Date, or PascalCase class/enum name). Null for enum literals.")
    visibility: Literal["public", "private", "protected", "package"] = Field(default="public", description="UML visibility")
    isDerived: bool = Field(default=False, description="Whether this is a derived/computed attribute.")
    defaultValue: Optional[str] = Field(default=None, description="Default value for the attribute.")
    isOptional: bool = Field(default=False, description="Whether this attribute is optional/nullable.")


class MethodSpec(BaseModel):
    name: str = Field(min_length=1, max_length=50, description="Method name in camelCase only (e.g. getName, calculateTotal). No parameters or return type here.")
    returnType: str = Field(default="void", description="Return type only (e.g. str, int, void). No colon prefix.")
    visibility: Literal["public", "private", "protected", "package"] = Field(default="public", description="UML visibility")
    parameters: List[MethodParameterSpec] = Field(default_factory=list, description="Method parameters, empty if none")
    isAbstract: bool = Field(default=False, description="Whether this is an abstract method.")
    implementationType: Literal["none", "code", "bal", "state_machine", "quantum_circuit"] = Field(
        default="none",
        description="Implementation type (e.g. none, code, bal, state_machine, quantum_circuit)."
    )
    code: Optional[str] = Field(default=None, description="Python implementation code for the method, including the full def statement.")


class SingleClassSpec(BaseModel):
    """A single UML class with attributes and optional methods."""
    className: str = Field(min_length=1, max_length=30, description="Class name in PascalCase, ONE word only (e.g. User, Order, Payment)")
    attributes: List[AttributeSpec] = Field(default_factory=list, description="Class attributes.")
    methods: List[MethodSpec] = Field(default_factory=list, description="Class methods for core domain behavior.")
    isAbstract: bool = Field(default=False, description="Whether this is an abstract class.")
    isEnumeration: bool = Field(default=False, description="Whether this is an enumeration.")


class RelationshipSpec(BaseModel):
    type: Literal[
        "Association", "Inheritance", "Composition", "Aggregation",
        "Realization", "Dependency",
    ] = Field(default="Association", description="Relationship type (e.g. Association, Inheritance, Composition, Aggregation).")
    source: str = Field(description="Source class name")
    target: str = Field(description="Target class name")
    sourceMultiplicity: str = Field(default="1", description="Source multiplicity: 1, 0..1, 0..*, or 1..*")
    targetMultiplicity: str = Field(default="*", description="Target multiplicity: 1, 0..1, 0..*, or 1..*")
    name: Optional[str] = Field(default=None, description="Optional relationship name")


class SystemClassSpec(BaseModel):
    """A complete class diagram with multiple classes and relationships."""
    systemName: str = Field(default="", description="Descriptive system name")
    classes: List[SingleClassSpec] = Field(min_length=1, description="All classes in the system.")
    relationships: List[RelationshipSpec] = Field(default_factory=list, description="Relationships between classes.")


# -- Modification schemas --

# Hallucinated "placeholder" tokens the LLM invents for required name fields it
# has no real value for (e.g. add_class.target.className). Matched
# case-insensitively as a substring so any of these anywhere in the name flags
# it as junk. Covers the live cases "...ClassNamePlaceholderHere" and
# "ChatbotHandlerClassNamePlaceholder".
# NOTE: substring-matched, so keep these specific enough not to collide with a
# legitimate domain class name (e.g. a real "Todo" class). Avoid bare tokens
# like "todo"/"tbd"/"xxx".
_PLACEHOLDER_TOKENS = (
    "placeholder",
    "classnamehere",
    "classname here",
    "namehere",
    "yourclassname",
    "yourclass",
    "<name>",
    "<class>",
    "<classname>",
    "enterclassname",
    "insertclassname",
    "exampleclassname",
    "newclassname",
)


def _is_placeholder(value: str | None) -> bool:
    """Return True if *value* looks like a hallucinated placeholder token.

    Case-insensitive substring match against ``_PLACEHOLDER_TOKENS`` so a leak
    like ``RolePermissionAssociationClassNamePlaceholderHere`` or
    ``ChatbotHandlerClassNamePlaceholder`` is caught even when the LLM prefixes
    or suffixes it with a real-looking word.
    """
    if not value or not isinstance(value, str):
        return False
    low = value.strip().lower()
    return any(tok in low for tok in _PLACEHOLDER_TOKENS)


def _clean_name(value: str | None) -> str | None:
    """Strip JSON artifacts (}, ], etc.) and null out hallucinated placeholders.

    First removes trailing JSON syntax the LLM may leak into a name, then nulls
    the value entirely if it matches a placeholder token — so a leak can never
    reach the applied model or the success message.
    """
    if not value:
        return value
    cleaned = re.sub(r'[{}\[\],]+$', '', value).strip() or None
    if _is_placeholder(cleaned):
        return None
    return cleaned


class ClassModificationTarget(BaseModel):
    className: Optional[str] = Field(default=None, description="Target class name")
    attributeName: Optional[str] = Field(default=None, description="Target attribute name within the class")
    methodName: Optional[str] = Field(default=None, description="Target method name within the class")
    sourceClass: Optional[str] = Field(default=None, description="Source class for relationship modifications")
    targetClass: Optional[str] = Field(default=None, description="Target class for relationship modifications")

    @model_validator(mode='after')
    def strip_json_artifacts(self) -> 'ClassModificationTarget':
        """Remove trailing JSON syntax artifacts from name fields."""
        self.className = _clean_name(self.className)
        self.attributeName = _clean_name(self.attributeName)
        self.methodName = _clean_name(self.methodName)
        self.sourceClass = _clean_name(self.sourceClass)
        self.targetClass = _clean_name(self.targetClass)
        return self


class ClassModificationChanges(BaseModel):
    name: Optional[str] = Field(default=None, max_length=30, description="New name for rename operations (PascalCase, ONE word only)")
    type: Optional[str] = Field(default=None, description="New type for attribute/parameter changes")
    visibility: Optional[Literal["public", "private", "protected", "package"]] = None
    returnType: Optional[str] = None
    parameters: Optional[List[MethodParameterSpec]] = None
    relationshipType: Optional[Literal[
        "Association", "Inheritance", "Composition", "Aggregation",
        "Realization", "Dependency",
    ]] = None
    sourceMultiplicity: Optional[str] = None
    targetMultiplicity: Optional[str] = None
    className: Optional[str] = Field(default=None, max_length=30, description="Class name in PascalCase for add_class action (ONE word only, e.g. User, Order)")
    attributes: Optional[List[AttributeSpec]] = Field(default=None, description="Attributes for add_class action")
    methods: Optional[List[MethodSpec]] = Field(default=None, description="Methods for add_class action")
    isDerived: Optional[bool] = Field(default=None, description="Set derived status for attribute")
    defaultValue: Optional[str] = Field(default=None, description="Set default value for attribute")
    isOptional: Optional[bool] = Field(default=None, description="Set optional status for attribute")
    isAbstract: Optional[bool] = Field(default=None, description="Set abstract status for class")
    implementationType: Optional[Literal["none", "code", "bal", "state_machine", "quantum_circuit"]] = Field(default=None, description="Implementation type for method.")
    code: Optional[str] = Field(default=None, description="Python code for method implementation")
    isEnumeration: Optional[bool] = Field(default=None, description="Set enumeration status for class")

    @field_validator('name', 'className', mode='before')
    @classmethod
    def _null_placeholder_names(cls, v):
        """Null a hallucinated placeholder name BEFORE length validation.

        These fields cap at 30 chars; a long leak like
        "ChatbotHandlerClassNamePlaceholder" (34 chars) would otherwise raise a
        max_length ValidationError and fail the whole modification. Running in
        ``mode='before'`` nulls it first so the constraint never sees the junk.
        """
        if isinstance(v, str) and _is_placeholder(v):
            return None
        return v

    @model_validator(mode='after')
    def strip_json_artifacts(self) -> 'ClassModificationChanges':
        """Strip JSON artifacts and null any hallucinated placeholder name.

        Also drops placeholder-named attributes/methods (their ``name`` field
        has min_length=1 so it can't be nulled in place — the whole entry is
        removed instead) so a leak can't survive on a sub-element either.
        """
        self.name = _clean_name(self.name)
        self.className = _clean_name(self.className)
        if self.attributes:
            self.attributes = [a for a in self.attributes if not _is_placeholder(a.name)]
        if self.methods:
            self.methods = [m for m in self.methods if not _is_placeholder(m.name)]
        return self


class ClassModification(BaseModel):
    action: Literal[
        "add_class", "modify_class",
        "add_attribute", "modify_attribute",
        "add_method", "modify_method",
        "add_relationship", "modify_relationship",
        "remove_element",
        "extract_class", "split_class", "merge_classes",
        "promote_attribute", "add_enum",
    ] = Field(description="Action to perform.")
    target: ClassModificationTarget
    changes: Optional[ClassModificationChanges] = Field(default=None, description="Changes to apply. Required for all actions except remove_element.")

    @model_validator(mode='after')
    def resolve_add_class_name(self) -> 'ClassModification':
        """For add_class, source the real class name and clear junk target.

        The new class name belongs in ``changes.className``; ``target.className``
        is meaningless for add_class (the target is a brand-new class). The LLM
        frequently hallucinates a placeholder there. This validator:

        1. Resolves the real, non-placeholder name from whichever of
           ``changes.className`` / ``target.className`` actually holds it
           (self-consistency: if the name landed only in target, promote it).
        2. Always clears ``target.className`` so a placeholder can never leak
           into the applied model or the success message.

        Field-level cleaning has already nulled obvious placeholders by the time
        this runs, so anything surviving here is treated as a real name.
        """
        if self.action != "add_class":
            return self

        # changes is created lazily so add_class always has somewhere to write
        if self.changes is None:
            self.changes = ClassModificationChanges()

        # Both are already placeholder-cleaned (None if junk). Prefer the
        # canonical location (changes.className); fall back to target.className.
        resolved = self.changes.className or _clean_name(self.target.className)
        self.changes.className = resolved

        # target.className is meaningless for a NEW class — never let it through.
        self.target.className = None
        return self


class ClassModificationResponse(BaseModel):
    modifications: List[ClassModification] = Field(min_length=1, description="List of modifications to apply")
