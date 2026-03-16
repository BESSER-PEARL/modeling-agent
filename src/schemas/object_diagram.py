"""Pydantic schemas for Object Diagram structured outputs."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ObjectAttributeSpec(BaseModel):
    name: str
    value: str
    attributeId: Optional[str] = None


class SingleObjectSpec(BaseModel):
    """Schema for a single object instance."""
    objectName: str = Field(min_length=1)
    className: str = Field(min_length=1)
    classId: Optional[str] = None
    attributes: List[ObjectAttributeSpec] = Field(default_factory=list)


class ObjectLinkSpec(BaseModel):
    source: str
    target: str
    relationshipType: Optional[str] = None


class SystemObjectSpec(BaseModel):
    """Schema for a complete object diagram system."""
    systemName: str = ""
    objects: List[SingleObjectSpec] = Field(min_length=1)
    links: List[ObjectLinkSpec] = Field(default_factory=list)


# -- Modification schemas --

class ObjectModificationTarget(BaseModel):
    objectName: Optional[str] = None
    attributeName: Optional[str] = None
    sourceObject: Optional[str] = None
    targetObject: Optional[str] = None

class ObjectModificationChanges(BaseModel):
    objectName: Optional[str] = None
    value: Optional[str] = None
    relationshipType: Optional[str] = None

class ObjectModification(BaseModel):
    action: str
    target: ObjectModificationTarget
    changes: Optional[ObjectModificationChanges] = None

class ObjectModificationResponse(BaseModel):
    modifications: List[ObjectModification] = Field(min_length=1)
