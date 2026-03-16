"""Pydantic schemas for GUI NoCode Diagram structured outputs."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class GUISampleDataPoint(BaseModel):
    name: str
    value: Any = 0
    color: Optional[str] = None


class GUISectionSpec(BaseModel):
    type: Literal[
        "hero", "feature_list", "content", "form", "table",
        "bar_chart", "pie_chart", "line_chart", "radar_chart",
        "dashboard", "metric_card", "stats_grid", "footer",
        "two_column",
    ] = "content"
    title: str = ""
    body: Optional[str] = None
    items: List[str] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)
    ctaLabel: Optional[str] = None
    className: Optional[str] = None
    sampleData: List[GUISampleDataPoint] = Field(default_factory=list)


class SingleGUIElementSpec(BaseModel):
    """Schema for a single GUI element (page with one section)."""
    pageName: str = Field(min_length=1)
    section: GUISectionSpec


class GUIPageSpec(BaseModel):
    pageName: str = Field(min_length=1)
    sections: List[GUISectionSpec] = Field(default_factory=list)


class SystemGUISpec(BaseModel):
    """Schema for a complete GUI system with multiple pages."""
    systemName: str = ""
    pages: List[GUIPageSpec] = Field(min_length=1)


# -- Modification schema --

class GUIModificationSpec(BaseModel):
    """Schema for GUI diagram modification operations."""
    operation: Literal["append_section", "rename_page", "remove_page"] = "append_section"
    pageName: str = Field(min_length=1)
    newPageName: Optional[str] = None
    section: Optional[GUISectionSpec] = None
