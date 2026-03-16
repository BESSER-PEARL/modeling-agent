"""Pydantic schemas for Class Diagram structured outputs."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MethodParameterSpec(BaseModel):
    name: str
    type: str = "String"


class AttributeSpec(BaseModel):
    name: str = Field(min_length=1)
    type: str = "String"
    visibility: Literal["public", "private", "protected", "package"] = "public"


class MethodSpec(BaseModel):
    name: str = Field(min_length=1)
    returnType: str = "void"
    visibility: Literal["public", "private", "protected", "package"] = "public"
    parameters: List[MethodParameterSpec] = Field(default_factory=list)


class SingleClassSpec(BaseModel):
    """Schema for a single class element."""
    className: str = Field(min_length=1)
    attributes: List[AttributeSpec] = Field(default_factory=list)
    methods: List[MethodSpec] = Field(default_factory=list)


class RelationshipSpec(BaseModel):
    type: Literal[
        "Association", "Inheritance", "Composition", "Aggregation",
        "Realization", "Dependency",
    ] = "Association"
    source: str
    target: str
    sourceMultiplicity: str = "1"
    targetMultiplicity: str = "*"
    name: Optional[str] = None


class SystemClassSpec(BaseModel):
    """Schema for a complete class diagram system."""
    systemName: str = ""
    classes: List[SingleClassSpec] = Field(min_length=1)
    relationships: List[RelationshipSpec] = Field(default_factory=list)


# -- Modification schemas --

class ClassModificationTarget(BaseModel):
    className: Optional[str] = None
    attributeName: Optional[str] = None
    methodName: Optional[str] = None
    sourceClass: Optional[str] = None
    targetClass: Optional[str] = None


class ClassModificationChanges(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    visibility: Optional[Literal["public", "private", "protected", "package"]] = None
    returnType: Optional[str] = None
    parameters: Optional[List[MethodParameterSpec]] = None
    relationshipType: Optional[str] = None
    sourceMultiplicity: Optional[str] = None
    targetMultiplicity: Optional[str] = None


class ClassModification(BaseModel):
    action: str
    target: ClassModificationTarget
    changes: Optional[ClassModificationChanges] = None


class ClassModificationResponse(BaseModel):
    modifications: List[ClassModification]
