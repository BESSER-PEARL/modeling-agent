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
        description="Optional CSS color for rendering this data point (e.g. '#FF5733', 'red')",
    )


class GUISectionSpec(BaseModel):
    type: Literal[
        "hero", "feature_list", "content", "form", "table",
        "bar_chart", "pie_chart", "line_chart", "radar_chart",
        "dashboard", "metric_card", "stats_grid", "footer",
        "two_column",
    ] = Field(
        default="content",
        description=(
            "Section layout type. "
            "'hero': prominent banner with title, body, and CTA. "
            "'feature_list': bulleted list of features using items. "
            "'content': generic text section with title and body. "
            "'form': input form whose field names come from fields. "
            "'table': data table whose column headers come from fields and row data from sampleData. "
            "'bar_chart': vertical bar chart rendered from sampleData. "
            "'pie_chart': pie/donut chart rendered from sampleData. "
            "'line_chart': line chart rendered from sampleData. "
            "'radar_chart': radar/spider chart rendered from sampleData. "
            "'dashboard': composite overview combining metric cards and charts. "
            "'metric_card': single KPI card showing a value and label. "
            "'stats_grid': grid of metric cards rendered from sampleData. "
            "'footer': page footer with links and text. "
            "'two_column': side-by-side two-column layout."
        ),
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
        description="List of display strings used by feature_list (bullet items), dashboard (widget labels), or similar list-oriented sections",
    )
    fields: List[str] = Field(
        default_factory=list,
        description="List of field/column names used by form (input labels) and table (column headers) sections",
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
        description="Sample data points for chart and stats sections (bar_chart, pie_chart, line_chart, radar_chart, stats_grid, table)",
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
        description="Modification operation: 'append_section' adds a new section to a page, 'rename_page' changes a page name, 'remove_page' deletes a page",
    )
    pageName: str = Field(
        min_length=1,
        description="Name of the target page to modify",
    )
    newPageName: Optional[str] = Field(
        default=None,
        description="New page name (only used with 'rename_page' operation)",
    )
    section: Optional[GUISectionSpec] = Field(
        default=None,
        description="Section to append (only used with 'append_section' operation)",
    )
