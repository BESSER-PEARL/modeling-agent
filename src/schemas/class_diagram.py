"""Pydantic schemas for Class Diagram structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import re

from pydantic import BaseModel, Field, model_validator


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

def _clean_name(value: str | None) -> str | None:
    """Strip JSON artifacts (},  ],  etc.) that the LLM may include in names."""
    if not value:
        return value
    return re.sub(r'[{}\[\],]+$', '', value).strip() or None


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

    @model_validator(mode='after')
    def strip_json_artifacts(self) -> 'ClassModificationChanges':
        self.name = _clean_name(self.name)
        if self.className:
            self.className = _clean_name(self.className)
        return self
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


class ClassModificationResponse(BaseModel):
    modifications: List[ClassModification] = Field(min_length=1, description="List of modifications to apply")
