"""Pydantic schemas for GUI NoCode Diagram structured outputs."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class GUISampleDataPoint(BaseModel):
    name: str = Field(
        description="Label for the data point (e.g. category name, x-axis label)",
    )
    # NOTE: This must be a CONCRETE type. ``Any`` renders a typeless JSON-schema
    # property which OpenAI strict structured-output mode rejects with a 400
    # BadRequestError — that single field broke EVERY GUI modification call
    # (GUIModificationSpec embeds GUISectionSpec embeds this model). A concrete
    # Union of str/int/float renders as a clean ``anyOf`` that strict mode
    # accepts while still allowing both numeric and label values.
    value: Optional[Union[str, int, float]] = Field(
        default=0,
        description="Numeric or string value for the data point (e.g. 42).",
    )
    color: Optional[str] = Field(
        default=None,
        description="Optional CSS color for this data point.",
    )


class GUIStatItem(BaseModel):
    """A single stat card (label + displayed value) for a ``stats_grid``.

    Concrete-typed so it survives OpenAI strict structured output — see the
    module note above about why ``Any``/open-dict fields are forbidden here.
    """
    label: str = Field(
        default="",
        description="Stat card label, e.g. 'Total Users'.",
    )
    value: str = Field(
        default="",
        description="Displayed stat value as a string, e.g. '1,234' or '87%'.",
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
    stats: List[GUIStatItem] = Field(
        default_factory=list,
        description=(
            "Stat cards for a 'stats_grid' section, each a {label, value} pair "
            "(e.g. [{\"label\": \"Total Users\", \"value\": \"1,234\"}]). Prefer "
            "this over 'items' for stats_grid so the displayed figures survive."
        ),
    )
    # Recursive sub-sections for a "two_column" layout. Concrete self-typed
    # (not Any/dict) so OpenAI strict structured output still validates — a
    # typeless field here would 400 EVERY GUI generate/modification call.
    left: Optional["GUISectionSpec"] = Field(
        default=None,
        description="Nested section rendered in the LEFT column of a 'two_column' layout.",
    )
    right: Optional["GUISectionSpec"] = Field(
        default=None,
        description="Nested section rendered in the RIGHT column of a 'two_column' layout.",
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
    """Schema for GUI diagram modification operations.

    Covers the common edit requests: adding/removing pages and sections,
    renaming them, recoloring a section or the whole page, and reordering
    a section to the top/bottom of its page.
    """
    operation: Literal[
        "append_section",
        "remove_section",
        "rename_page",
        "add_page",
        "remove_page",
        "rename_section",
        "recolor_section",
        "recolor_page",
        "reorder_section",
    ] = Field(
        default="append_section",
        description=(
            "Operation to perform. Use rename_page to rename a page, "
            "rename_section to rename/retitle a section, recolor_section or "
            "recolor_page to change colors, reorder_section to move a section "
            "up/down, append_section/remove_section to add or delete a section, "
            "add_page/remove_page to add or delete a page."
        ),
    )
    pageName: str = Field(
        min_length=1,
        description="Name of the target page to modify (or the page to act on)",
    )
    newPageName: Optional[str] = Field(
        default=None,
        max_length=80,
        description="New page name (used with 'rename_page' and 'add_page').",
    )
    sectionTitle: Optional[str] = Field(
        default=None,
        description=(
            "Current title/heading or type of the target section to find "
            "(used with rename_section, recolor_section, remove_section, "
            "reorder_section). May be a heading like 'Recent edits' or a type "
            "like 'hero'/'table'."
        ),
    )
    newSectionTitle: Optional[str] = Field(
        default=None,
        description="New title/heading for the section (used with 'rename_section').",
    )
    color: Optional[str] = Field(
        default=None,
        description=(
            "Target color as a CSS value or common name (e.g. 'red', '#ff0000') "
            "for recolor_section / recolor_page operations."
        ),
    )
    position: Optional[Literal["top", "bottom", "up", "down"]] = Field(
        default=None,
        description="Where to move the section for 'reorder_section'.",
    )
    section: Optional[GUISectionSpec] = Field(
        default=None,
        description="Section to append (used with 'append_section' / 'add_page').",
    )


# ``GUISectionSpec`` references itself (left/right) via forward refs — resolve
# them now that every referenced model is defined.
GUISectionSpec.model_rebuild()
