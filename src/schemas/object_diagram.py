"""Pydantic schemas for Object Diagram structured outputs."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ObjectAttributeSpec(BaseModel):
    name: str = Field(
        description="Attribute name matching the reference class definition."
    )
    value: str = Field(
        description="Concrete example value for this attribute (e.g. 'John Doe', '99.99')."
    )
    attributeId: Optional[str] = Field(
        default=None,
        description="Element id of this attribute in the reference class diagram."
    )


class SingleObjectSpec(BaseModel):
    """Schema for a single object instance."""
    objectName: str = Field(
        min_length=1,
        description="Instance name in lowerCamelCase (e.g. 'user1', 'orderA')."
    )
    className: str = Field(
        min_length=1,
        description="Class this object instantiates, matching the reference class diagram."
    )
    classId: Optional[str] = Field(
        default=None,
        description="Element id of the class in the reference class diagram."
    )
    attributes: List[ObjectAttributeSpec] = Field(
        default_factory=list,
        description="Attribute name/value pairs for this object instance."
    )


class ObjectLinkSpec(BaseModel):
    source: str = Field(
        description="Source object name."
    )
    target: str = Field(
        description="Target object name."
    )
    relationshipType: Optional[str] = Field(
        default=None,
        description="Relationship type or name (e.g. 'association', 'placedBy')."
    )


class SystemObjectSpec(BaseModel):
    """Schema for a complete object diagram system."""
    systemName: str = Field(
        default="",
        description="Descriptive name for the object diagram."
    )
    objects: List[SingleObjectSpec] = Field(
        min_length=1,
        description="Object instances in the diagram."
    )
    links: List[ObjectLinkSpec] = Field(
        default_factory=list,
        description="Links between objects representing relationships."
    )


# -- Modification schemas --

class ObjectModificationTarget(BaseModel):
    objectName: Optional[str] = Field(
        default=None,
        description="Object name to modify or remove."
    )
    attributeName: Optional[str] = Field(
        default=None,
        description="Attribute name to modify on the target object."
    )
    sourceObject: Optional[str] = Field(
        default=None,
        description="Source object name for link operations."
    )
    targetObject: Optional[str] = Field(
        default=None,
        description="Target object name for link operations."
    )

class ObjectModificationChanges(BaseModel):
    objectName: Optional[str] = Field(
        default=None,
        description="New or renamed object name in lowerCamelCase."
    )
    className: Optional[str] = Field(
        default=None,
        description="Class name for add_object."
    )
    attributes: Optional[List[ObjectAttributeSpec]] = Field(
        default=None,
        description="Attributes with concrete values for add_object."
    )
    value: Optional[str] = Field(
        default=None,
        description="New attribute value to set."
    )
    relationshipType: Optional[str] = Field(
        default=None,
        description="Relationship type for a new or modified link."
    )

class ObjectModification(BaseModel):
    action: str = Field(
        description="Action to perform (e.g. add_object, modify_attribute_value, add_link, remove_element)."
    )
    target: ObjectModificationTarget = Field(
        description="Identifies the element to modify."
    )
    changes: Optional[ObjectModificationChanges] = Field(
        default=None,
        description="Changes to apply. Not needed for remove_element."
    )

class ObjectModificationResponse(BaseModel):
    modifications: List[ObjectModification] = Field(
        min_length=1,
        description="List of modifications to apply to the object diagram."
    )
