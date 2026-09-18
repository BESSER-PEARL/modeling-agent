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


class GUITableRow(BaseModel):
    """One sample row for a data table.

    ``cells`` is a flat list of display strings aligned positionally to the
    table's ``columns`` (cell *i* belongs under column *i*). Kept as a concrete
    ``List[str]`` — never an open dict — so OpenAI strict structured output
    still validates (see the ``GUISampleDataPoint`` note above).
    """
    cells: List[str] = Field(
        default_factory=list,
        description="Cell values for this row as strings, one per column, in column order.",
    )


class GUIBindSpec(BaseModel):
    """Data-binding spec for a DATA section (Phase 3).

    A DATA section pairs LLM-authored HTML *chrome* (carrying a
    ``<!--WIDGET:slot-->`` placeholder) with a structured binding the server
    turns into a real, recognizer-compatible widget node via the typed
    builders. Every field is CONCRETE-typed so the model still survives OpenAI
    strict structured output — see the ``GUISampleDataPoint`` note above about
    why ``Any``/open-dict fields are forbidden (a single one 400s EVERY GUI
    generate/modification call).
    """
    kind: Literal[
        "table", "bar_chart", "pie_chart", "line_chart", "radar_chart",
        "metric_card", "form", "dashboard",
    ] = Field(
        description="Which typed data widget to bind (e.g. table, bar_chart, form).",
    )
    className: Optional[str] = Field(
        default=None,
        description="Reference class name from the ClassDiagram for data binding.",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Column / field names for table and form widgets.",
    )
    series: List[str] = Field(
        default_factory=list,
        description="Series names for chart widgets.",
    )
    sampleData: List[GUISampleDataPoint] = Field(
        default_factory=list,
        description="Sample data points for the bound widget preview.",
    )
    rows: List[GUITableRow] = Field(
        default_factory=list,
        description=(
            "Sample rows for a 'table' binding — 4-6 realistic rows, each a list "
            "of cell strings aligned to 'columns'. REQUIRED for tables so the "
            "table renders with real data instead of an empty grid."
        ),
    )


class GUISectionSpec(BaseModel):
    # Phase 3: ``type`` is now OPTIONAL — a section may instead be authored as
    # rich HTML (``html``) or bound to a typed widget (``bind``). Kept as a
    # concrete Optional[Literal] (never Any) so strict structured output still
    # validates. When absent, the legacy typed builder defaults to "content".
    type: Optional[Literal[
        "hero", "feature_list", "content", "form", "table",
        "bar_chart", "pie_chart", "line_chart", "radar_chart",
        "dashboard", "metric_card", "stats_grid", "footer",
        "two_column",
    ]] = Field(
        default=None,
        description="Legacy section layout type (hero, content, form, table, ...). Optional; prefer 'html' or 'bind'.",
    )
    html: Optional[str] = Field(
        default=None,
        description=(
            "LLM-authored rich themed HTML for this section using the .ds-* "
            "component classes and semantic tags. MUST start with a heading and "
            "carry a stable class on its root element. For a DATA section, embed "
            "a <!--WIDGET:slot--> comment where the bound widget should splice in."
        ),
    )
    bind: Optional["GUIBindSpec"] = Field(
        default=None,
        description="Structured data binding for a DATA section (paired with optional 'html' chrome).",
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
    domain: Optional[str] = Field(
        default=None,
        description=(
            "Design domain driving the visual theme: one of government, finance, "
            "health, startup, default."
        ),
    )
    pages: List[GUIPageSpec] = Field(
        min_length=1,
        description="List of pages in the GUI system (at least one required)",
    )


# -- Complete-system wire schema (Phase 3 authoring) --

class AuthoredGUISectionSpec(BaseModel):
    """One authored section of a generated page — the structured mirror of the
    authoring prompt's two shapes: rich themed HTML, or a data binding with
    optional HTML chrome. Both fields optional so OpenAI strict structured
    output validates; the handler drops a section carrying neither.
    """
    html: Optional[str] = Field(
        default=None,
        description=(
            "Rich themed HTML for this section using ONLY the .ds-* design-"
            "system classes and semantic tags (h1-h6, p, span, a, ul, li, "
            "button, img with data: URI or no src, svg, plain table markup). "
            "MUST start with a heading and carry a stable class on its root "
            "element. No <script>/<style>, no external URLs, no lorem ipsum. "
            "For a DATA section, embed a <!--WIDGET:kind--> comment where the "
            "bound widget splices in."
        ),
    )
    bind: Optional[GUIBindSpec] = Field(
        default=None,
        description=(
            "Structured data binding for a DATA section (a live table/chart/"
            "form/metric widget). Populate it — columns+rows for tables, "
            "sampleData for charts — or the widget renders empty."
        ),
    )


class AuthoredGUIPageSpec(BaseModel):
    name: str = Field(
        min_length=1,
        description="Unique display name for this page (e.g. 'Home', 'Bookings').",
    )
    sections: List[AuthoredGUISectionSpec] = Field(
        default_factory=list,
        description="Ordered sections that make up this page.",
    )


class GUIThemeSpec(BaseModel):
    """Custom design-token overrides — the escape hatch from the preset themes.

    When the user asks for a specific look ('dark mode', 'our brand green',
    'pastel and playful'), fill only the tokens that should change; the full
    .ds-* stylesheet is regenerated from the merged tokens, so the result
    stays coherent. Omit entirely to use the domain preset.
    """
    primary: Optional[str] = Field(default=None, description="Primary brand color (CSS color).")
    secondary: Optional[str] = Field(default=None, description="Secondary color (CSS color).")
    accent: Optional[str] = Field(default=None, description="Accent color (CSS color).")
    background: Optional[str] = Field(default=None, description="Page background color.")
    surface: Optional[str] = Field(default=None, description="Card/surface color.")
    text: Optional[str] = Field(default=None, description="Main text color.")
    muted: Optional[str] = Field(default=None, description="Muted/secondary text color.")
    border: Optional[str] = Field(default=None, description="Border color.")
    radius: Optional[str] = Field(default=None, description="Corner radius (e.g. '0px', '12px').")
    heroBackground: Optional[str] = Field(
        default=None,
        description="Hero band background (color or CSS gradient).",
    )
    heroText: Optional[str] = Field(default=None, description="Hero band text color.")


class AuthoredSystemGUISpec(BaseModel):
    """Wire schema for complete-GUI generation (structured output).

    Replaces the former free-text JSON contract 1:1 (same keys), so the
    downstream page assembly is untouched — but the output is now schema-
    enforced: no truncated JSON, no parse-and-salvage, no malformed specs.
    """
    projectName: str = Field(default="App", description="Application name.")
    domain: Optional[Literal[
        "government", "finance", "health", "startup", "default",
    ]] = Field(
        default=None,
        description="Design domain driving the visual theme.",
    )
    theme: Optional[GUIThemeSpec] = Field(
        default=None,
        description=(
            "Custom design-token overrides for a user-requested look. Fill "
            "ONLY when the request implies a specific style the domain preset "
            "doesn't deliver; omit otherwise."
        ),
    )
    css: Optional[str] = Field(
        default=None,
        description=(
            "OPTIONAL app-level stylesheet for your own custom classes "
            "(prefix them app-) used in section html — plain CSS with class "
            "rules and @media blocks only. No @import, no url(), no "
            "webfonts, no scripts. Use it for gradients, feature layouts, "
            "hover states — real visual identity beyond the ds-* kit."
        ),
    )
    pages: List[AuthoredGUIPageSpec] = Field(
        min_length=1,
        description="Pages of the app (at least one).",
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
        "edit_section",
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
            "edit_section to REWRITE an existing section's content in place "
            "(target via sectionTitle, replacement in 'section'), "
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
        description=(
            "Section payload (used with 'append_section' / 'edit_section' / "
            "'add_page'). Prefer the authored shapes: 'html' (rich themed "
            ".ds-* markup) or 'bind' (a populated data widget, optionally "
            "with 'html' chrome around a <!--WIDGET:kind--> slot)."
        ),
    )


class GUIModificationBatchSpec(BaseModel):
    """A user request may bundle several edits ('rename Home to Overview and
    make the hero red') — the batch applies them in order, each one a
    :class:`GUIModificationSpec`.
    """
    operations: List[GUIModificationSpec] = Field(
        min_length=1,
        max_length=5,
        description=(
            "The edit operations to apply, in order — ONE per distinct edit "
            "the user asked for. Most requests need exactly one."
        ),
    )


# ``GUISectionSpec`` references itself (left/right) and ``GUIBindSpec`` via
# forward refs — resolve them now that every referenced model is defined.
GUIBindSpec.model_rebuild()
GUISectionSpec.model_rebuild()
