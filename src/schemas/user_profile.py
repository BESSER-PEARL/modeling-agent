"""Pydantic schemas for User Profile (UserDiagram) structured outputs.

A user-profile model describes a *target user* as a set of class-instance
boxes drawn from the fixed user metamodel (User, Personal_Information,
Competence, Language, …).  Unlike an object diagram, each attribute row is a
*matching criterion* carrying a comparison operator (``age >= 18``,
``level == B2``) rather than a plain instance value.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# Comparison operators supported by the editor's UserModelAttribute element.
UserProfileOperator = Literal["<", "<=", "==", ">=", ">"]


class UserProfileAttributeSpec(BaseModel):
    name: str = Field(
        description="Attribute name matching the metamodel class definition (e.g. 'age')."
    )
    operator: UserProfileOperator = Field(
        default="==",
        description=(
            "Comparison operator expressing the matching criterion. Infer from "
            "phrasing: 'older than 18' -> '>', 'at least B2' -> '>=', "
            "'exactly Spanish' -> '=='. Default to '=='."
        ),
    )
    value: str = Field(
        description="Concrete criterion value (e.g. '18', 'Spanish', 'B2')."
    )
    attributeId: Optional[str] = Field(
        default=None,
        description="Element id of this attribute in the metamodel (populated server-side).",
    )
    type: Optional[str] = Field(
        default=None,
        description="Attribute type from the metamodel (populated server-side).",
    )


class SingleUserProfileSpec(BaseModel):
    """Schema for a single user-profile class instance (one box)."""

    className: str = Field(
        min_length=1,
        max_length=40,
        description="Metamodel class this box instantiates (e.g. 'Personal_Information', 'Language').",
    )
    profileName: Optional[str] = Field(
        default=None,
        description="Instance identifier for link wiring (e.g. 'personal_information1'). Optional.",
    )
    classId: Optional[str] = Field(
        default=None,
        description="Element id of the class in the metamodel (populated server-side).",
    )
    attributes: List[UserProfileAttributeSpec] = Field(
        default_factory=list,
        description="Attribute matching criteria for this profile box.",
    )


class UserProfileLinkSpec(BaseModel):
    source: str = Field(description="Source profileName (or className).")
    target: str = Field(description="Target profileName (or className).")
    relationshipType: Optional[str] = Field(
        default=None,
        description="Relationship name (e.g. 'speaks', 'hasCompetence').",
    )


class SystemUserProfileSpec(BaseModel):
    """Schema for a complete user-profile model."""

    systemName: str = Field(
        default="",
        description="Descriptive name for the user profile.",
    )
    profiles: List[SingleUserProfileSpec] = Field(
        min_length=1,
        description="Profile class instances in the model.",
    )
    links: List[UserProfileLinkSpec] = Field(
        default_factory=list,
        description="Links between profile boxes representing metamodel relationships.",
    )


# -- Modification schemas --
# Reuses the ObjectDiagram action vocabulary so the frontend UserDiagram
# modifier and TS ModelModification union need no new action strings.


class UserProfileModificationTarget(BaseModel):
    profileName: Optional[str] = Field(
        default=None,
        description="Profile box (instance) name to modify or remove.",
    )
    className: Optional[str] = Field(
        default=None,
        description="Class name of the profile box (alternative identifier).",
    )
    attributeName: Optional[str] = Field(
        default=None,
        description="Attribute name to modify on the target profile box.",
    )
    sourceProfile: Optional[str] = Field(
        default=None,
        description="Source profile name for link operations.",
    )
    targetProfile: Optional[str] = Field(
        default=None,
        description="Target profile name for link operations.",
    )


class UserProfileModificationChanges(BaseModel):
    profileName: Optional[str] = Field(
        default=None,
        max_length=40,
        description="New or renamed profile box (instance) name.",
    )
    className: Optional[str] = Field(
        default=None,
        max_length=40,
        description="Metamodel class name for add_object.",
    )
    classId: Optional[str] = Field(
        default=None,
        description="Element id of the metamodel class (populated server-side).",
    )
    icon: Optional[str] = Field(
        default=None,
        description="Metamodel class icon SVG for add_object (populated server-side).",
    )
    attributes: Optional[List[UserProfileAttributeSpec]] = Field(
        default=None,
        description="Attribute criteria for add_object.",
    )
    operator: Optional[UserProfileOperator] = Field(
        default=None,
        description="New comparison operator for modify_attribute_value.",
    )
    value: Optional[str] = Field(
        default=None,
        description="New attribute criterion value to set.",
    )
    relationshipType: Optional[str] = Field(
        default=None,
        description="Relationship type for a new or modified link.",
    )


class UserProfileModification(BaseModel):
    action: Literal[
        "add_object",
        "modify_object",
        "modify_attribute_value",
        "add_link",
        "remove_element",
    ] = Field(description="Action to perform.")
    target: UserProfileModificationTarget = Field(
        description="Identifies the element to modify."
    )
    changes: Optional[UserProfileModificationChanges] = Field(
        default=None,
        description="Changes to apply. Not needed for remove_element.",
    )


class UserProfileModificationResponse(BaseModel):
    modifications: List[UserProfileModification] = Field(
        min_length=1,
        description="List of modifications to apply to the user-profile model.",
    )
