"""Pydantic schemas for GUI NoCode Diagram structured outputs."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class GUISampleDataPoint(BaseModel):
    name: str = Field(
        description="Label for the data point (e.g. category name, x-axis label)",
    )
    value: Any = Field(
        default=0,
        description="Numeric or string value for the data point",
    )
    color: Optional[str] = Field(
        default=None,
        description="Optional CSS color for this data point.",
    )


class GUISectionSpec(BaseModel):
    type: Literal[
        "hero", "feature_list", "content", "form", "table",
        "bar_chart", "pie_chart", "line_chart", "radar_chart",
        "dashboard", "metric_card", "stats_grid", "footer",
        "two_column",
    ] = Field(
        default="content",
        description="Section layout type (e.g. hero, content, form, table, bar_chart, dashboard, footer).",
    )
    title: str = Field(
        default="",
        description="Heading text displayed at the top of the section",
    )
    body: Optional[str] = Field(
        default=None,
        description="Body/paragraph text content for the section",
    )
    items: List[str] = Field(
        default_factory=list,
        description="Display strings for list-oriented sections (e.g. feature_list, dashboard).",
    )
    fields: List[str] = Field(
        default_factory=list,
        description="Field or column names for form and table sections.",
    )
    ctaLabel: Optional[str] = Field(
        default=None,
        description="Call-to-action button label (e.g. 'Sign Up', 'Learn More')",
    )
    className: Optional[str] = Field(
        default=None,
        description="Reference class name from the ClassDiagram for data binding",
    )
    sampleData: List[GUISampleDataPoint] = Field(
        default_factory=list,
        description="Sample data points for chart, stats, and table sections.",
    )


class SingleGUIElementSpec(BaseModel):
    """Schema for a single GUI element (page with one section)."""
    pageName: str = Field(
        min_length=1,
        description="Name of the page this element belongs to",
    )
    section: GUISectionSpec = Field(
        description="The GUI section to add to the page",
    )


class GUIPageSpec(BaseModel):
    pageName: str = Field(
        min_length=1,
        description="Unique display name for this page",
    )
    sections: List[GUISectionSpec] = Field(
        default_factory=list,
        description="Ordered list of sections that make up this page",
    )


class SystemGUISpec(BaseModel):
    """Schema for a complete GUI system with multiple pages."""
    systemName: str = Field(
        default="",
        description="Name of the GUI application or system",
    )
    pages: List[GUIPageSpec] = Field(
        min_length=1,
        description="List of pages in the GUI system (at least one required)",
    )


# -- Modification schema --

class GUIModificationSpec(BaseModel):
    """Schema for GUI diagram modification operations."""
    operation: Literal["append_section", "rename_page", "remove_page"] = Field(
        default="append_section",
        description="Operation: append_section, rename_page, or remove_page.",
    )
    pageName: str = Field(
        min_length=1,
        description="Name of the target page to modify",
    )
    newPageName: Optional[str] = Field(
        default=None,
        max_length=50,
        description="New page name (only used with 'rename_page' operation)",
    )
    section: Optional[GUISectionSpec] = Field(
        default=None,
        description="Section to append (only used with 'append_section' operation)",
    )
