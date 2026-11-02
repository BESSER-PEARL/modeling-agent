"""GUI (no-code) prompts: authoring a themed app, its reasoning pass, interactions, mockups and modification."""

from typing import Any, Dict, List, Optional


# A cheap, always-valid, CSP-safe inline-SVG placeholder the LLM can reuse (and
# recolor) so cards/heroes are never image-less. Kept tiny so it can be inlined
# many times without bloating the payload. Survives the converter's SVG whitelist
# (svg/rect/circle) untouched.
SVG_THUMB_HINT = (
    "<svg viewBox='0 0 80 80' width='72' height='72' role='img' aria-label='thumbnail' "
    "style='border-radius:12px'>"
    "<rect width='80' height='80' rx='12' fill='#e2e8f0'/>"
    "<circle cx='40' cy='30' r='14' fill='#94a3b8'/>"
    "<rect x='16' y='50' width='48' height='18' rx='9' fill='#cbd5e1'/></svg>"
)


def interactions_block(has_classes: bool) -> str:
    """How authored markup states link, navigation and method intent."""
    method_rule = (
        "\n- Class-method buttons: <button data-method='methodName' data-class='ClassName'> "
        "using ONLY methods listed in the class diagram; place them next to that class's "
        "table (the method runs on the selected row). Every method is also added under "
        "its class's table automatically, so only place one where it adds value."
        if has_classes else ""
    )
    form_rule = (
        "- A form that creates a record: <form data-class='ClassName'> with one input "
        "per attribute a new record needs, each <input name='attribute_name'> carrying "
        "the exact attribute name, with a <label>. Search, login and contact forms get "
        "no data-class.\n"
        if has_classes else ""
    )
    return (
        "INTERACTIONS — every link and button must DO something in the generated app:\n"
        "- Page links: <a href='/route'> where route is the target page's name lowercased "
        "with spaces replaced by hyphens (page 'Front Desk' -> '/front-desk'). Never "
        "href='#' and never an external URL.\n"
        "- Navigation buttons: <button data-page='Page Name'>."
        f"{method_rule}\n"
        "- Create / edit / delete buttons act through X's data table: 'Add X' / 'New X' "
        "opens the table's add dialog (on this page, or the page holding X's table); "
        "'Edit X' / 'Delete X' act on the row selected in X's table, so place them on a "
        "page with that table. No data-page on these.\n"
        f"{form_rule}"
        "- No action-less buttons (filters, sort toggles): use a styled span or badge."
    )


def mockup_block(image_count: int) -> str:
    """Authoring rules when the user attached screenshots or mockups (#643)."""
    screens = "the attached image" if image_count == 1 else f"the {image_count} attached images"
    return (
        f"REFERENCE MOCKUP — the user attached {screens} of the screen(s) to build. The "
        "mockup is the authority: where it conflicts with any design guidance above, "
        "follow the mockup.\n"
        "- One page per image, in the order given, named after what the screen shows. Do "
        "not add pages or sections the mockup does not show; build every section it does "
        "show, top to bottom, in the same order and layout (columns, card grids, "
        "sidebars, alignment).\n"
        "- Copy every visible text exactly: headings, paragraphs, labels, buttons, menu "
        "items, table cells, footer lines. Keep bold and italic words. Never rewrite, "
        "shorten or translate the copy.\n"
        "- Match the look in your \"css\": colours as hex values read from the image, "
        "font sizes and weights, letter-spacing, uppercase labels, spacing, border radius, "
        "borders and shadows. Serif text gets font-family 'Fraunces', Georgia, serif; sans "
        "text 'Inter', sans-serif. Emit a \"theme\" with the mockup's own colours.\n"
        "- A band that spans the full page width (top bar, full-bleed hero, footer) "
        "carries ds-nav, ds-hero or ds-footer on its root element, restyled in your css "
        "to match the mockup; any other section is placed in a centred 1200px column. "
        "ds-nav is already a flex row with space-between: put the bar's items directly "
        "inside it, with no inner wrapper div.\n"
        "- Photos, illustrations and logos become an inline <svg> that approximates the "
        "picture's shapes and colours; keep any caption drawn over it.\n"
        "- A table or chart of records in the mockup is a DATA section: reproduce its "
        "columns and visible rows as the widget's columns/rows or sampleData, and bind "
        "it to the matching class when a class diagram is given.\n"
        "- A menu entry for a screen you are not building stays visible as drawn, as a "
        "styled span."
    )


def system_prompt(class_block: str) -> str:
    return f"""You are a UI modeling expert for a no-code web editor.

Return ONLY JSON with this shape:
{{
  "pageName": "Home",
  "section": {{
    "type": "hero|feature_list|content|form|table|bar_chart|pie_chart|line_chart|radar_chart|dashboard|metric_card|stats_grid|footer|two_column",
    "title": "Section title",
    "body": "Optional descriptive text",
    "items": ["Optional list item / footer link label"],
    "fields": ["Optional field label"],
    "ctaLabel": "Optional button label",
    "className": "Optional class name from Class Diagram to bind data to",
    "sampleData": [
      {{"name": "Realistic label from domain", "value": 42}}
    ],
    "stats": [{{"label": "Total Users", "value": "1,234"}}],
    "left": {{"type": "...", "title": "..."}},
    "right": {{"type": "...", "title": "..."}}
  }}
}}

Section types:
- hero: Hero/landing banner with title, body, CTA button
- feature_list: List of feature items
- content: Generic text section
- form: Input form with fields
- table: Data table bound to a class (requires className)
- bar_chart / pie_chart / line_chart / radar_chart: Chart visualisations bound to a class
- dashboard: Combined table + charts for a class
- metric_card: Single KPI metric card from a class
- stats_grid: Row of stat cards. Provide \"stats\" as [{{\"label\": \"Total Users\", \"value\": \"1,234\"}}] — put the figure in \"value\"
- footer: Page footer with project name and links. Provide items as link labels
- two_column: Side-by-side layout. Provide "left" and "right" as nested section specs

Rules:
1. Keep content concise and practical.
2. Use section type that best matches user request.
3. When the user mentions data, statistics, or visualisation, prefer chart/table/dashboard types.
4. When a className is provided or classes are available, bind data sections to them.
5. For table/chart/dashboard sections, ALWAYS include a "sampleData" array with 4-6 realistic preview rows.
6. Return JSON only.{class_block}"""


def complete_system_prompt(domain: str, exemplars_block: str, interactions_block: str, class_block: str, mockup_block: str) -> str:
    return f"""You are a senior product designer AUTHORING a themed, production-realistic web app for the **{domain}** domain.

You do NOT pick sections from a fixed widget menu. You AUTHOR the markup for each section using a shared design system, and you BIND real data widgets where the app shows data.

Return ONLY JSON with this shape:
{{
  "projectName": "App name",
  "domain": "{domain}",
  "css": "YOUR stylesheet — you own the visual design of this app (see DESIGN OWNERSHIP below)",
  "pages": [
    {{
      "name": "Home",
      "sections": [ <ordered list of sections, each one of the two shapes below> ]
    }}
  ]
}}

Each section is EXACTLY ONE of these two shapes:

(a) HTML section — rich themed markup you write yourself:
{{
  "html": "<section class='ds-section'><h2 class='ds-heading'>Section title</h2> ...rich content using .ds-* classes + semantic tags... </section>"
}}
   - It MUST start with a heading (h1-h6) and its ROOT element MUST carry a stable class (e.g. ds-section, ds-hero, ds-footer).
   - YOU are the designer. Write your own markup with your own class names, styled in the top-level "css" stylesheet — the .ds-* utility classes below are optional helpers (fastest for widget chrome), never a limit. Allowed tags: h1-h6, p, span, a, div, section, header, footer, ul, li, button, img (data-URI or no src), svg, plain table markup, form/label/input. Inline style attributes are fine for one-off accents.
   - NO <script>/<style>, NO external URLs, NO webfonts (blocked by CSP). NO lorem ipsum, NO empty placeholders.

(b) DATA section — LLM-authored chrome + a real, live, data-bound widget:
{{
  "bind": {{
    "kind": "table|bar_chart|pie_chart|line_chart|radar_chart|metric_card|form|dashboard",
    "className": "Entity name from the class diagram (when available)",
    "title": "Specific widget title, e.g. 'Overdue loans'",
    "labelField": "chart: category attribute", "valueField": "chart/metric: numeric attribute",
    "aggregation": "chart/metric: count|sum|avg|min|max",
    "columns": ["Column", "headers"],
    "rows": [{{"cells": ["cell A1", "cell A2"]}}, {{"cells": ["cell B1", "cell B2"]}}],
    "series": ["series names"],
    "sampleData": [{{"name": "Realistic label", "value": 42}}]
  }},
  "html": "<section class='ds-section'><h3 class='ds-heading'>Title</h3><div class='ds-card'><!--WIDGET:table--></div></section>"
}}
   - Put a <!--WIDGET:kind--> comment inside the chrome where the widget belongs; the server splices the real, data-bound widget there.
   - The "html" chrome is OPTIONAL — omit it and the widget is card-wrapped automatically — but authoring chrome around it gives a far nicer result.
   - Whatever widget you place, populate it or it renders empty: a table needs "columns" + "rows" ("cells" aligned 1:1 to "columns"); a chart needs "sampleData" (name + numeric value; pie adds a "color" hex).
   - Give every widget a specific "title" (never 'Data Table' / 'Bar Chart'). With a class diagram, set a chart's "labelField" (category attribute) and its "aggregation": the chart groups records by labelField and shows per label the "count" of records (no valueField needed) or the "sum"/"avg"/"min"/"max" of "valueField" (numeric attribute) — e.g. count of books by genre, sum of amount by month; omit "aggregation" only to plot valueField of each record, labelled by a unique field. A metric_card shows the "count" of records (no valueField), or the "sum"/"avg"/"min"/"max" of its "valueField" — set its "aggregation" to match its title (e.g. 'Active members' = count, 'Average fine' = avg of fine).
   - Pick the widget that fits the content: a table suits tabular records (orders, bookings, transactions, inventory); a card grid (below) usually reads better for listings of people / products / profiles / features. Use your judgment.

Utility classes, pre-styled for the {domain} theme (OPTIONAL helpers — reach for them for speed and widget chrome, not as your design ceiling):
  ds-page, ds-container, ds-section, ds-hero, ds-heading, ds-card, ds-grid-2, ds-grid-3,
  ds-kpi, ds-kpi-value, ds-kpi-label, ds-table-wrap, ds-btn, ds-btn-primary, ds-notice,
  ds-badge, ds-field, ds-label, ds-input, ds-nav, ds-footer.

Proven, editable-safe block patterns for the {domain} domain — COMPOSE from these and REUSE their .ds-* classes:
{exemplars_block}

DESIGN OWNERSHIP — you are a world-class product designer and the look of this app is YOURS. The {domain} preset is only the floor:
- When the user asks for a specific look (dark mode, brand colors, pastel, corporate navy, ...), emit a top-level "theme" object overriding ONLY the tokens that must change (primary/secondary/accent/background/surface/text/muted/border/radius/heroBackground/heroText) — the whole design system re-derives from them coherently.
- Otherwise OMIT "theme" entirely and let the domain preset style everything.
- Author the top-level "css" stylesheet for EVERY app: a complete design — layout systems, gradients, hover/focus states, responsive rules — plain CSS only (class rules + @media; no @import/url()/webfonts). Your class names are yours (avoid the ds- prefix). The auto-added navigation header carries the class assistant-nav-header — restyle it in your css if your design wants a different top bar. The ONLY predefined CSS variables are the --ds-* tokens (var(--ds-primary), var(--ds-accent), var(--ds-border), ...) — never reference variables that do not exist. Typefaces available: 'Inter' (UI) and 'Fraunces' (display serif) — you may use font-family with these names in your css.

Realism directives (this is what makes the result credible, not generic):
- Reproduce the SPECIFIC real-world artifact the request implies, with its real structure. E.g. a government service page has an official header, an eligibility notice, an applicant-identity fieldset, a multi-step application form, a document upload, declarations/consent checkboxes, and a submit + application tracker — NOT a generic marketing page.
- Write plausible, concrete domain copy: real-sounding labels, figures and names. Never leave a placeholder.

CARD GRID (a strong pattern for listings of people / products / profiles / features / plans):
  <section class='ds-section'><h2 class='ds-heading'>Featured Profiles</h2>
    <div class='ds-grid-3'>
      <div class='ds-card'>{SVG_THUMB_HINT}<h3 class='ds-heading'>Ava, 27</h3><p>Loves hiking & jazz. 92% match.</p><a class='ds-btn ds-btn-primary' href='/profiles'>View</a></div>
      ...one card per item, each with a real name/label + concrete copy + (optionally) a visual...
    </div></section>

{interactions_block}

Imagery (use it where it strengthens the design — heroes, profile/product cards, feature icons):
- ONLY an inline <svg ...>...</svg> or an <img> whose src is a data: URI ever renders — external/http image URLs are blocked by CSP, so never use them.
- Reusable, CSP-safe placeholder if you want one: {SVG_THUMB_HINT}

Design judgment — build what THIS request actually needs; do not pad or force a fixed skeleton:
1. Pages: let the request set the count. A multi-screen app / platform / dashboard naturally spans several pages (an overview/home plus a screen per major feature); a "landing page" or single-"page" request is one focused page. Each page a coherent set of ordered sections.
2. Lead each page with the section that fits its purpose — a marketing ds-hero for a landing page; a data screen with a heading + KPIs or a ds-notice. Don't default to a hero for a data app.
3. Full-width sections (ds-hero, ds-footer, ds-nav) span the page; other sections sit in cards inside a centered container.
4. Close with a ds-footer when it suits the page (most marketing/landing pages); an app/data screen may not need one.
5. Populate every widget you place, and give each page enough real copy to read as finished — but include only what the request calls for, not filler.
6. VISUAL HIERARCHY: open every page with a page-header moment (title + one-line subtitle + the page's primary action). Vary the rhythm — full-width bands, grids, cards, accent moments — generous whitespace; NEVER a monotone stack of identical white cards. Commit ONE signature visual moment per page (a gradient band, an oversized stat, a colored panel) — safe-but-flat everywhere is a failure.
7. SAMPLE DATA REALISM: figures must be plausible and VARIED — a chart where every value is 1 (or several series with identical data) is a FAILURE. Write numbers a real business would show; one meaningful series unless comparing series is the point. A section whose heading promises a chart / distribution / trend / breakdown MUST be a DATA section with a chart bind carrying that varied sampleData — never a table and never plain html. Headings are never generic labels like 'Table' or 'Section'.
8. Keep an entity's field names IDENTICAL across every section it appears in.
9. Return JSON only.{class_block}{mockup_block}"""


def data_model_step_for_classes(class_metadata_text: str) -> str:
    return (
        "1. DATA MODEL (fixed). The app is built on this class diagram — "
        "do NOT invent entities, fields or methods; use these exact names:\n"
        f"{class_metadata_text}\n"
        "Decide which page shows each class, and where each class method "
        "is triggered (a button next to that class's table).\n"
    )


CHART_RULE_FOR_CLASSES = (
    "A chart groups records by one attribute (labelField) and "
    "shows per group the count of records or the sum/avg/min/max "
    "of ONE numeric attribute (valueField); a metric card shows a "
    "count, sum, average, minimum or maximum over all records."
)


DATA_MODEL_STEP = (
    "1. DATA MODEL FIRST. Identify the 2-4 core entities this app "
    "revolves around. For EACH entity write its exact field names + "
    "types, e.g. Book: title:str, author:str, genre:str, year:int, "
    "available:bool. These field names are the contract for the whole "
    "UI — pick them once and do not vary them.\n"
)


CHART_RULE = "A chart aggregates ONE field (e.g. count of books by genre)."


def reasoning_prompt(user_request: str, data_model_step: str, chart_rule: str, domain: str) -> str:
    return (
        "You are a UI/UX design expert. Think step by step about the "
        "following web application request and plan the page layout.\n\n"
        f"User Request: {user_request}\n\n"
        "Work in this order:\n"
        f"{data_model_step}"
        "2. Pages: decide how many screens THIS request needs — a full "
        "app/platform/dashboard spans several (an overview/home plus a "
        "screen per major feature or entity); a 'landing page' or single "
        "'page' request is one focused page. Don't pad.\n"
        "3. Sections per page: choose the app TYPE (marketing / dashboard / "
        "data-catalog / CRUD) and lead each page with the section that fits "
        "it — do NOT default to a marketing hero for a data app.\n"
        "4. Binding (most important): every table/chart/form that shows an "
        "entity MUST reuse the EXACT field names from step 1 — the same "
        f"entity must look identical on every page. {chart_rule}\n"
        "5. Navigation flow — for EVERY link and button, the page it opens "
        "or the method it runs — and realistic domain sample data built "
        "from the step-1 fields.\n"
        "6. DOMAIN: classify this app as ONE of government / finance / health / "
        "startup / default — this sets the visual theme AND the tone of the copy "
        f"(our heuristic guess is '{domain}'; confirm it or correct it). Then "
        "commit to reproducing the SPECIFIC real-world artifact the request "
        "implies, with its real structure and concrete, plausible copy — never "
        "generic placeholder text.\n\n"
        "Design a modern, clean UI. Think like a Lovable/Vercel designer — "
        "clean typography, generous spacing, purposeful color, and data that "
        "stays coherent across every page."
    )


MOCKUP_REASONING_ADDENDUM = (
    "\n\nThe attached image(s) are mockups of the screens to build and "
    "override the steps above: first list every section of each screen, "
    "top to bottom, with its exact text, layout and colours (hex), then "
    "plan the pages and bindings that reproduce them. No extra pages or "
    "sections."
)


def modification_prompt(outline: str, class_metadata: Optional[List[Dict[str, Any]]], class_block: str) -> str:
    return f"""You are the design agent for an existing web GUI. Apply the user's requested edit(s) precisely and surgically — change ONLY what they asked for.

CURRENT APP (pages with their sections, in order):
{outline}

Emit 1-5 operations — ONE per distinct edit in the request (most requests need exactly one):
- rename_page / add_page / remove_page
- rename_section / recolor_section / recolor_page / reorder_section / remove_section
- append_section — add a NEW section to a page (author it in "section")
- edit_section — REWRITE an existing section's content in place: target it with "sectionTitle" (its heading as listed above) and author the full replacement in "section"

For append_section / edit_section / add_page, author "section" in the app's design system:
  PREFERRED — rich themed HTML: {{"html": "<section class='ds-section'><h2 class='ds-heading'>Title</h2> ...concrete content... </section>"}}
   - Root element carries a stable class; use ONLY .ds-* classes + semantic tags (h1-h6, p, span, a, ul, li, button, svg, plain tables). No scripts/styles/external URLs. Real copy, never placeholders.
  DATA — a live bound widget: {{"bind": {{"kind": "table|bar_chart|pie_chart|line_chart|radar_chart|metric_card|form|dashboard", "className": "Entity from the class diagram", "columns": [...], "rows": [{{"cells": [...]}}], "sampleData": [{{"name": "...", "value": 42}}]}}, "html": "<section class='ds-section'><h3 class='ds-heading'>Title</h3><div class='ds-card'><!--WIDGET:kind--></div></section>"}}
   - Populate the widget (columns+rows for tables, sampleData for charts) or it renders empty.
Design-system classes: ds-section, ds-hero, ds-heading, ds-card, ds-grid-2, ds-grid-3, ds-kpi, ds-kpi-value, ds-kpi-label, ds-table-wrap, ds-btn, ds-btn-primary, ds-notice, ds-badge, ds-field, ds-label, ds-input, ds-container, ds-footer.

{interactions_block(bool(class_metadata))}

Rules:
1. Choose the most specific operation — never append_section for a rename/recolor/reorder/remove/edit request.
2. Use page names and section headings EXACTLY as listed above.
3. edit_section replaces the WHOLE section: carry over what should stay, change what was asked.
4. Keep an entity's field names identical to how they appear elsewhere in the app.{class_block}"""
