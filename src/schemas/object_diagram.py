"""Pydantic schemas for Object Diagram structured outputs."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ObjectAttributeSpec(BaseModel):
    name: str = Field(
        description="Exact attribute name from the reference class definition (without visibility or type suffix)."
    )
    value: str = Field(
        description="A realistic, concrete example value for this attribute (not a type name). E.g. 'John Doe', '99.99', 'true'."
    )
    attributeId: Optional[str] = Field(
        default=None,
        description="The exact element id of this attribute in the reference class diagram. Must match the reference if one is provided."
    )


class SingleObjectSpec(BaseModel):
    """Schema for a single object instance."""
    objectName: str = Field(
        min_length=1,
        description="Instance name in lowerCamelCase, e.g. 'user1', 'orderA'. Must start with a lowercase letter."
    )
    className: str = Field(
        min_length=1,
        description="The class this object instantiates. Must exactly match a class name from the reference class diagram when one is provided."
    )
    classId: Optional[str] = Field(
        default=None,
        description="The exact element id of the class in the reference class diagram. Required when a reference diagram is provided."
    )
    attributes: List[ObjectAttributeSpec] = Field(
        default_factory=list,
        description="List of attribute name/value pairs. Must include ALL attributes defined in the reference class, using exact names and ids from the reference."
    )


class ObjectLinkSpec(BaseModel):
    source: str = Field(
        description="The objectName of the source object in this link. Must match an existing object's objectName."
    )
    target: str = Field(
        description="The objectName of the target object in this link. Must match an existing object's objectName."
    )
    relationshipType: Optional[str] = Field(
        default=None,
        description="Name or type of the relationship, e.g. 'association', 'placedBy'. Should match a relationship from the reference class diagram when available."
    )


class SystemObjectSpec(BaseModel):
    """Schema for a complete object diagram system."""
    systemName: str = Field(
        default="",
        description="A short descriptive name for the object diagram, e.g. 'LibraryScenario'."
    )
    objects: List[SingleObjectSpec] = Field(
        min_length=1,
        description="List of 3-6 related object instances. Each must reference a class from the reference class diagram when one is provided."
    )
    links: List[ObjectLinkSpec] = Field(
        default_factory=list,
        description="Links between objects representing relationships. Source and target must reference objectNames from the objects list."
    )


# -- Modification schemas --

class ObjectModificationTarget(BaseModel):
    objectName: Optional[str] = Field(
        default=None,
        description="The objectName of the object to modify or remove. Must match an existing object in the current diagram."
    )
    attributeName: Optional[str] = Field(
        default=None,
        description="The attribute name to modify on the target object. Must match an attribute defined on that object."
    )
    sourceObject: Optional[str] = Field(
        default=None,
        description="For link operations: the objectName of the source object."
    )
    targetObject: Optional[str] = Field(
        default=None,
        description="For link operations: the objectName of the target object."
    )

class ObjectModificationChanges(BaseModel):
    objectName: Optional[str] = Field(
        default=None,
        description="New objectName when renaming an object. Must be lowerCamelCase, e.g. 'user2'."
    )
    value: Optional[str] = Field(
        default=None,
        description="New attribute value to set. Must be a concrete value, not a type."
    )
    relationshipType: Optional[str] = Field(
        default=None,
        description="Relationship type for a new or modified link, e.g. 'association', 'owns'."
    )

class ObjectModification(BaseModel):
    action: str = Field(
        description="The modification action: 'modify_object', 'modify_attribute_value', 'add_link', or 'remove_element'."
    )
    target: ObjectModificationTarget = Field(
        description="Identifies the element to modify. Use objectName for object operations, sourceObject/targetObject for link operations."
    )
    changes: Optional[ObjectModificationChanges] = Field(
        default=None,
        description="The new values to apply. Not needed for 'remove_element' action."
    )

class ObjectModificationResponse(BaseModel):
    modifications: List[ObjectModification] = Field(
        min_length=1,
        description="One or more modifications to apply to the object diagram. Use a single entry for one change, multiple entries for batch changes."
    )
