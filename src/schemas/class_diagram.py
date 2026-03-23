"""Pydantic schemas for Class Diagram structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MethodParameterSpec(BaseModel):
    name: str = Field(description="Parameter name in camelCase")
    type: str = Field(default="String", description="Parameter type: String, int, boolean, float, Date, or a custom class name")


class AttributeSpec(BaseModel):
    name: str = Field(min_length=1, description="Attribute name in camelCase")
    type: Optional[str] = Field(default=None, description="Attribute type: String, int, boolean, float, Date, a custom class name, or an enumeration name (e.g., 'OrderStatus'). For ENUMERATION VALUES, leave this null/empty — enum literals have no type. When referencing an enum as an attribute type in a regular class, use the enum's exact PascalCase name. Never use UUID, long, decimal, BigDecimal, LocalDate, List, or Set.")
    visibility: Literal["public", "private", "protected", "package"] = Field(default="public", description="UML visibility")
    isDerived: bool = Field(default=False, description="Whether this is a derived/computed attribute. Rendered with '/' prefix in UML notation.")
    defaultValue: Optional[str] = Field(default=None, description="Default value for the attribute. Rendered as '= value' suffix.")
    isOptional: bool = Field(default=False, description="Whether this attribute is optional/nullable. Rendered with '?' suffix.")


class MethodSpec(BaseModel):
    name: str = Field(min_length=1, description="Method name in camelCase. Only include core domain methods, never getters/setters.")
    returnType: str = Field(default="void", description="Return type")
    visibility: Literal["public", "private", "protected", "package"] = Field(default="public", description="UML visibility")
    parameters: List[MethodParameterSpec] = Field(default_factory=list, description="Method parameters, empty if none")
    isAbstract: bool = Field(default=False, description="Whether this is an abstract method (no implementation).")
    implementationType: Literal["none", "code", "bal", "state_machine", "quantum_circuit"] = Field(
        default="none",
        description="Implementation type: 'none' for UML-only, 'code' for Python code, 'bal' for BESSER Action Language, 'state_machine' to link a state machine, 'quantum_circuit' to link a quantum circuit."
    )
    code: Optional[str] = Field(default=None, description="Python implementation code for the method. Include the full def statement. Example: 'def calculate_total(self):\\n    return sum(item.price for item in self.items)'")


class SingleClassSpec(BaseModel):
    """A single UML class with attributes and optional methods."""
    className: str = Field(min_length=1, description="Class name in PascalCase")
    attributes: List[AttributeSpec] = Field(default_factory=list, description="Class attributes. Include all relevant domain attributes (IDs, timestamps, status fields).")
    methods: List[MethodSpec] = Field(default_factory=list, description="Class methods. Only include if explicitly requested or core domain behavior (e.g. withdraw, calculateTotal). Never include getters/setters.")
    isAbstract: bool = Field(default=False, description="Whether this is an abstract class. Rendered with italic name and <<abstract>> stereotype.")
    isEnumeration: bool = Field(default=False, description="Whether this is an enumeration. Attributes become enum values (no types needed).")


class RelationshipSpec(BaseModel):
    type: Literal[
        "Association", "Inheritance", "Composition", "Aggregation",
        "Realization", "Dependency",
    ] = Field(default="Association", description="Relationship type: Association (general), Inheritance (is-a), Composition (strong has-a, part dies with whole), Aggregation (weak has-a), Realization (interface), Dependency")
    source: str = Field(description="Source class name")
    target: str = Field(description="Target class name")
    sourceMultiplicity: str = Field(default="1", description="Source multiplicity: 1, 0..1, 0..*, or 1..*")
    targetMultiplicity: str = Field(default="*", description="Target multiplicity: 1, 0..1, 0..*, or 1..*")
    name: Optional[str] = Field(default=None, description="Optional relationship name")


class SystemClassSpec(BaseModel):
    """A complete class diagram with multiple classes and relationships."""
    systemName: str = Field(default="", description="Descriptive system name")
    classes: List[SingleClassSpec] = Field(min_length=1, description="All classes in the system. Each should have 3-5+ attributes.")
    relationships: List[RelationshipSpec] = Field(default_factory=list, description="Relationships between classes. Always include multiplicities.")


# -- Modification schemas --

class ClassModificationTarget(BaseModel):
    className: Optional[str] = Field(default=None, description="Target class name")
    attributeName: Optional[str] = Field(default=None, description="Target attribute name within the class")
    methodName: Optional[str] = Field(default=None, description="Target method name within the class")
    sourceClass: Optional[str] = Field(default=None, description="Source class for relationship modifications")
    targetClass: Optional[str] = Field(default=None, description="Target class for relationship modifications")


class ClassModificationChanges(BaseModel):
    name: Optional[str] = Field(default=None, description="New name for rename operations")
    type: Optional[str] = Field(default=None, description="New type for attribute/parameter changes")
    visibility: Optional[Literal["public", "private", "protected", "package"]] = None
    returnType: Optional[str] = None
    parameters: Optional[List[MethodParameterSpec]] = None
    relationshipType: Optional[str] = None
    sourceMultiplicity: Optional[str] = None
    targetMultiplicity: Optional[str] = None
    className: Optional[str] = Field(default=None, description="Class name for add_class action")
    attributes: Optional[List[AttributeSpec]] = Field(default=None, description="Attributes for add_class action")
    methods: Optional[List[MethodSpec]] = Field(default=None, description="Methods for add_class action")
    isDerived: Optional[bool] = Field(default=None, description="Set derived status for attribute")
    defaultValue: Optional[str] = Field(default=None, description="Set default value for attribute")
    isOptional: Optional[bool] = Field(default=None, description="Set optional status for attribute")
    isAbstract: Optional[bool] = Field(default=None, description="Set abstract status for class")
    implementationType: Optional[str] = Field(default=None, description="Implementation type for method: none, code, bal, state_machine, quantum_circuit")
    code: Optional[str] = Field(default=None, description="Python code for method implementation")
    isEnumeration: Optional[bool] = Field(default=None, description="Set enumeration status for class")


class ClassModification(BaseModel):
    action: str = Field(description="Modification action: add_class, modify_class, add_attribute, modify_attribute, add_method, modify_method, add_relationship, modify_relationship, remove_element")
    target: ClassModificationTarget
    changes: Optional[ClassModificationChanges] = Field(default=None, description="The changes to apply. REQUIRED for all actions except remove_element. For modify_relationship, put the NEW values here (e.g. new sourceMultiplicity), not in target.")


class ClassModificationResponse(BaseModel):
    modifications: List[ClassModification] = Field(min_length=1, description="List of modifications to apply")
