"""
GUI No-Code Diagram Handler
Handles generation of GUINoCodeDiagram models for GrapesJS-based editor.
"""

from __future__ import annotations

import copy
import json
import re
import logging
from html import escape as _html_escape
from typing import Any, Dict, List, Optional

from ..core.base_handler import BaseDiagramHandler, LLMPredictionError
from .gui_html_converter import (
    html_to_components,
    find_widget_slots,
    replace_widget_slot,
)
from .gui_design_system import (
    stylesheet_rules,
    stylesheet_rules_from_tokens,
    merge_theme_overrides,
    theme_tokens,
    block_exemplars,
    pick_domain,
    _hex_to_rgba,
)
from model_config import MODEL_GENERATION_LARGE, MODEL_GENERATION_SMALL, MODEL_REASONING
from schemas import (
    AuthoredSystemGUISpec,
    GUIModificationBatchSpec,
    GUIModificationSpec,
    SingleGUIElementSpec,
)
from utilities.class_metadata import format_class_metadata_for_prompt

logger = logging.getLogger(__name__)

DEFAULT_GUI_VERSION = "0.21.13"

# Complete-system JSON is the one generation path that produces very large
# multi-page output. The shared free-text budget (LLM_MAX_TOKENS_LARGE = 8192)
# truncates ambitious requests mid-stream — exactly the requests users care
# about — collapsing the whole app to a one-line "Welcome" stub. Give this
# path a much larger cap. Scoped here (passed explicitly to predict_two_pass)
# so no other handler's budget changes.
GUI_COMPLETE_SYSTEM_MAX_TOKENS = 16384

# A cheap, always-valid, CSP-safe inline-SVG placeholder the LLM can reuse (and
# recolor) so cards/heroes are never image-less. Kept tiny so it can be inlined
# many times without bloating the payload. Survives the converter's SVG whitelist
# (svg/rect/circle) untouched.
_SVG_THUMB_HINT = (
    "<svg viewBox='0 0 80 80' width='72' height='72' role='img' aria-label='thumbnail' "
    "style='border-radius:12px'>"
    "<rect width='80' height='80' rx='12' fill='#e2e8f0'/>"
    "<circle cx='40' cy='30' r='14' fill='#94a3b8'/>"
    "<rect x='16' y='50' width='48' height='18' rx='9' fill='#cbd5e1'/></svg>"
)

# Chart colour palette (deterministic cycling)
_CHART_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085", "#d35400",
]

# Pie / radial-bar palette — matches the drag-and-drop editor defaults
_PIE_COLORS = ["#00C49F", "#0088FE", "#FFBB28", "#FF8042", "#A569BD"]


def _clean_text(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else fallback
    return fallback


def _coerce_numeric(value: Any) -> Any:
    """Best-effort convert a chart ``value`` to a number.

    ``GUISampleDataPoint.value`` is typed as ``str`` (so OpenAI strict
    structured outputs accept it), but charts expect numbers. Convert
    numeric-looking strings to int/float; leave non-numeric values as-is.
    """
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return value
        try:
            if "." in text or "e" in text.lower():
                return float(text)
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return value
    return value


def _sanitize_page_name(value: Any, fallback: str = "Page") -> str:
    label = _clean_text(value, fallback=fallback)
    if not label:
        return fallback
    label = re.sub(r"\s+", " ", label)
    return label[:40]


def _main_container(children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap a list of section components inside a centered <main> container."""
    return {
        "tagName": "main",
        "attributes": {"class": "assistant-main"},
        "style": {
            "max-width": "1200px",
            "margin": "0 auto",
            "padding": "24px 16px",
        },
        "components": children,
    }


_CSS_BLOCK_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_SAFE_SELECTOR_RE = re.compile(r"^[.#a-zA-Z][-\w\s.,:#>()\[\]='\"*+~^$|]*$")
_MAX_LLM_CSS_RULES = 160


def _parse_llm_css(css: Any) -> List[Dict[str, Any]]:
    """Parse an LLM-authored stylesheet into GrapesJS style-rule dicts.

    Supports plain rules and single-level @media blocks. The stylesheet is the
    LLM's creative escape hatch beyond the ds-* kit; safety comes first — one
    forbidden token (@import / url() / expression / javascript:) voids the
    WHOLE sheet (the ds baseline still styles the page), and malformed pieces
    are silently dropped, never raised.
    """
    rules: List[Dict[str, Any]] = []
    if not isinstance(css, str) or not css.strip():
        return rules
    text = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    if re.search(r"@import|url\s*\(|expression\s*\(|javascript:|</", text, re.IGNORECASE):
        logger.warning("[GUINoCode] LLM stylesheet rejected (forbidden token)")
        return rules

    def parse_decls(body: str) -> Dict[str, str]:
        style: Dict[str, str] = {}
        for decl in body.split(";"):
            if ":" not in decl:
                continue
            prop, _, value = decl.partition(":")
            prop, value = prop.strip(), value.strip()
            if prop and value and re.match(r"^[-a-zA-Z]+$", prop):
                style[prop] = value
        return style

    def add_rule(selector: str, body: str, media: Optional[str] = None) -> None:
        selector = selector.strip()
        if not selector or not _SAFE_SELECTOR_RE.match(selector):
            return
        style = parse_decls(body)
        if not style or len(rules) >= _MAX_LLM_CSS_RULES:
            return
        rule: Dict[str, Any] = {"selectors": [], "selectorsAdd": selector, "style": style}
        if media:
            rule["atRuleType"] = "media"
            rule["mediaText"] = media
        rules.append(rule)

    media_re = re.compile(r"@media([^{]+)\{((?:[^{}]*\{[^{}]*\})*)\s*\}", re.DOTALL)

    def handle_media(match: "re.Match") -> str:
        media_text = match.group(1).strip()
        for sel, body in _CSS_BLOCK_RE.findall(match.group(2)):
            add_rule(sel, body, media=media_text)
        return ""

    text = media_re.sub(handle_media, text)
    for sel, body in _CSS_BLOCK_RE.findall(text):
        if sel.strip().startswith("@"):
            continue
        add_rule(sel, body)
    return rules


def _default_wrapper_component(tokens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # The page canvas takes its ground from the THEME, not from constants —
    # a teal app must not sit on a foreign gray/Inter base.
    palette = (tokens or {}).get("palette", {})
    return {
        "type": "wrapper",
        "style": {
            "background-color": palette.get("background", "#f8fafc"),
            "font-family": (tokens or {}).get(
                "font", "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
            ),
            "color": palette.get("text", "#1e293b"),
            "min-height": "100vh",
        },
        "stylable": [
            "background",
            "background-color",
            "background-image",
            "background-repeat",
            "background-attachment",
            "background-position",
            "background-size",
        ],
        "components": [],
        "head": {"type": "head"},
        "docEl": {"tagName": "html"},
    }


def _default_gui_model() -> Dict[str, Any]:
    return {
        "pages": [
            {
                "name": "Home",
                "frames": [
                    {
                        "component": _default_wrapper_component(),
                    }
                ],
            }
        ],
        "styles": [],
        "assets": [],
        "symbols": [],
        "version": DEFAULT_GUI_VERSION,
    }


def _normalize_gui_model(candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return _default_gui_model()

    model = copy.deepcopy(candidate)
    pages = model.get("pages")
    if not isinstance(pages, list):
        pages = []
    model["pages"] = pages
    model["styles"] = model.get("styles") if isinstance(model.get("styles"), list) else []
    model["assets"] = model.get("assets") if isinstance(model.get("assets"), list) else []
    model["symbols"] = model.get("symbols") if isinstance(model.get("symbols"), list) else []
    model["version"] = model.get("version") if isinstance(model.get("version"), str) else DEFAULT_GUI_VERSION

    if not pages:
        model["pages"] = _default_gui_model()["pages"]

    return model


def _ensure_page_wrapper(page: Dict[str, Any]) -> Dict[str, Any]:
    frames = page.get("frames")
    if not isinstance(frames, list) or not frames:
        frames = [{"component": _default_wrapper_component()}]
        page["frames"] = frames

    first_frame = frames[0]
    if not isinstance(first_frame, dict):
        first_frame = {"component": _default_wrapper_component()}
        frames[0] = first_frame

    component = first_frame.get("component")
    if not isinstance(component, dict):
        component = _default_wrapper_component()
        first_frame["component"] = component

    components = component.get("components")
    if not isinstance(components, list):
        component["components"] = []

    return component


# ----------------------------------------------------------------------
# Page / section lookup + edit helpers (used by generate_modification)
# ----------------------------------------------------------------------

# Common CSS color names → hex, so requests like "make it red" map to a
# concrete value. Anything already looking like a CSS color is passed through.
_NAMED_COLORS = {
    "red": "#e74c3c", "blue": "#2563eb", "green": "#2ecc71",
    "yellow": "#f1c40f", "orange": "#e67e22", "purple": "#9b59b6",
    "pink": "#ff69b4", "teal": "#1abc9c", "cyan": "#00bcd4",
    "black": "#111827", "white": "#ffffff", "gray": "#6b7280",
    "grey": "#6b7280", "navy": "#1e3a5f", "indigo": "#4f46e5",
    "violet": "#8b5cf6", "brown": "#92400e", "gold": "#d4af37",
    "silver": "#c0c0c0", "maroon": "#800000", "lime": "#84cc16",
    "magenta": "#d946ef",
}


def _resolve_color(value: Any) -> Optional[str]:
    """Resolve a color name or CSS color to a concrete CSS color string."""
    text = _clean_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in _NAMED_COLORS:
        return _NAMED_COLORS[lowered]
    # Already a hex / rgb / hsl / named CSS value — pass through.
    if re.match(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", lowered) or lowered.startswith(("rgb", "hsl")):
        return text
    return text




def _iter_section_components(wrapper: Dict[str, Any]):
    """Yield ``(parent_list, index, component)`` for every top-level section.

    Sections live either directly under the page wrapper or one level deep
    inside an ``assistant-main`` container. This flattens both so callers can
    edit, remove, or reorder a section regardless of nesting.
    """
    top = wrapper.get("components")
    if not isinstance(top, list):
        return
    for comp in list(top):
        if not isinstance(comp, dict):
            continue
        cls = comp.get("attributes", {}).get("class", "") if isinstance(comp.get("attributes"), dict) else ""
        if cls == "assistant-main":
            inner = comp.get("components")
            if isinstance(inner, list):
                for child in list(inner):
                    if isinstance(child, dict):
                        yield inner, inner.index(child), child
        else:
            yield top, top.index(comp), comp


def _section_label(comp: Dict[str, Any]) -> str:
    """Human label for a section: its first heading text, else its type class."""
    def _first_heading(node: Dict[str, Any]) -> Optional[str]:
        for child in node.get("components", []) if isinstance(node.get("components"), list) else []:
            if not isinstance(child, dict):
                continue
            if child.get("tagName") in ("h1", "h2", "h3") and _clean_text(child.get("content")):
                return _clean_text(child.get("content"))
            nested = _first_heading(child)
            if nested:
                return nested
        return None

    heading = _first_heading(comp)
    if heading:
        return heading
    cls = comp.get("attributes", {}).get("class", "") if isinstance(comp.get("attributes"), dict) else ""
    return cls.replace("assistant-", "").replace("-", " ").strip() or "section"


def _set_section_heading(comp: Dict[str, Any], new_title: str) -> bool:
    """Replace the first heading's text in a section. Returns True if changed."""
    if not isinstance(comp.get("components"), list):
        return False
    for child in comp["components"]:
        if not isinstance(child, dict):
            continue
        if child.get("tagName") in ("h1", "h2", "h3"):
            child["content"] = new_title
            return True
        if _set_section_heading(child, new_title):
            return True
    return False


def _match_section(comp: Dict[str, Any], query: str) -> bool:
    """Does *comp* match the user's section reference (by heading or type)?"""
    q = _clean_text(query).lower()
    if not q:
        return False
    label = _section_label(comp).lower()
    if q in label or label in q:
        return True
    cls = comp.get("attributes", {}).get("class", "") if isinstance(comp.get("attributes"), dict) else ""
    section_type = cls.replace("assistant-", "")
    return bool(section_type) and q in section_type


def _hero_component(title: str, body: str, cta_label: str) -> Dict[str, Any]:
    return {
        "tagName": "section",
        "attributes": {"class": "assistant-hero"},
        "style": {
            "padding": "64px 48px",
            "background": "linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%)",
            "color": "#ffffff",
            "border-radius": "16px",
            "margin": "24px",
            "text-align": "center",
        },
        "components": [
            {
                "tagName": "h1",
                "content": title,
                "style": {
                    "margin": "0 0 16px 0",
                    "font-size": "2.25rem",
                    "font-weight": "800",
                    "letter-spacing": "-0.02em",
                    "line-height": "1.2",
                },
            },
            {
                "tagName": "p",
                "content": body,
                "style": {
                    "margin": "0 auto 28px auto",
                    "font-size": "1.1rem",
                    "line-height": "1.6",
                    "max-width": "600px",
                    "opacity": "0.9",
                },
            },
            {
                "tagName": "button",
                "content": cta_label,
                "attributes": {"class": "assistant-cta"},
                "style": {
                    "padding": "12px 28px",
                    "border": "none",
                    "border-radius": "10px",
                    "font-weight": "600",
                    "font-size": "1rem",
                    "background-color": "#ffffff",
                    "color": "#2563eb",
                    "cursor": "pointer",
                    "box-shadow": "0 2px 8px rgba(0,0,0,0.15)",
                },
            },
        ],
    }


def _feature_list_component(title: str, items: List[str]) -> Dict[str, Any]:
    cleaned_items = [item for item in (item.strip() for item in items if isinstance(item, str)) if item]
    if not cleaned_items:
        cleaned_items = ["Feature 1", "Feature 2", "Feature 3"]

    return {
        "tagName": "section",
        "attributes": {"class": "assistant-features"},
        "style": {
            "padding": "32px",
            "background-color": "#ffffff",
            "border-radius": "14px",
            "margin": "12px 24px",
            "box-shadow": "0 1px 4px rgba(0,0,0,0.06)",
            "border": "1px solid #f1f5f9",
        },
        "components": [
            {
                "tagName": "h2",
                "content": title,
                "style": {
                    "margin": "0 0 18px 0",
                    "font-size": "1.35rem",
                    "font-weight": "700",
                    "color": "#0f172a",
                },
            },
            {
                "tagName": "ul",
                "style": {"padding-left": "20px", "margin": "0"},
                "components": [
                    {
                        "tagName": "li",
                        "content": item,
                        "style": {
                            "margin": "10px 0",
                            "color": "#334155",
                            "line-height": "1.5",
                        },
                    }
                    for item in cleaned_items
                ],
            },
        ],
    }


def _content_component(title: str, body: str) -> Dict[str, Any]:
    return {
        "tagName": "section",
        "attributes": {"class": "assistant-content"},
        "style": {
            "padding": "32px",
            "background-color": "#ffffff",
            "border": "1px solid #f1f5f9",
            "border-radius": "14px",
            "margin": "12px 24px",
            "box-shadow": "0 1px 4px rgba(0,0,0,0.06)",
        },
        "components": [
            {
                "tagName": "h2",
                "content": title,
                "style": {
                    "margin": "0 0 12px 0",
                    "font-size": "1.35rem",
                    "font-weight": "700",
                    "color": "#0f172a",
                },
            },
            {
                "tagName": "p",
                "content": body,
                "style": {
                    "margin": "0",
                    "line-height": "1.6",
                    "color": "#475569",
                },
            },
        ],
    }


def _form_component(title: str, fields: List[str], cta_label: str) -> Dict[str, Any]:
    cleaned_fields = [field for field in (field.strip() for field in fields if isinstance(field, str)) if field]
    if not cleaned_fields:
        cleaned_fields = ["Name", "Email"]

    return {
        "tagName": "section",
        "attributes": {"class": "assistant-form"},
        "style": {
            "padding": "32px",
            "border": "1px solid #f1f5f9",
            "border-radius": "14px",
            "margin": "12px 24px",
            "background-color": "#ffffff",
            "box-shadow": "0 1px 4px rgba(0,0,0,0.06)",
        },
        "components": [
            {
                "tagName": "h2",
                "content": title,
                "style": {
                    "margin": "0 0 20px 0",
                    "font-size": "1.35rem",
                    "font-weight": "700",
                    "color": "#0f172a",
                },
            },
            {
                "tagName": "form",
                "components": [
                    {
                        "tagName": "div",
                        "style": {"display": "grid", "gap": "14px"},
                        "components": [
                            {
                                "tagName": "input",
                                "attributes": {
                                    "type": "text",
                                    "name": re.sub(r"[^a-z0-9_]+", "_", field.lower()),
                                    "placeholder": field,
                                },
                                "style": {
                                    "padding": "12px 14px",
                                    "border": "1px solid #e2e8f0",
                                    "border-radius": "10px",
                                    "font-size": "0.95rem",
                                    "background-color": "#f8fafc",
                                    "outline": "none",
                                },
                            }
                            for field in cleaned_fields
                        ],
                    },
                    {
                        "tagName": "button",
                        "content": cta_label,
                        "attributes": {"type": "button"},
                        "style": {
                            "margin-top": "16px",
                            "padding": "12px 24px",
                            "border": "none",
                            "border-radius": "10px",
                            "background-color": "#2563eb",
                            "color": "#ffffff",
                            "font-weight": "600",
                            "font-size": "0.95rem",
                            "cursor": "pointer",
                        },
                    },
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Data-bound component builders (charts, tables, dashboards)
# ---------------------------------------------------------------------------

def _resolve_class_binding(
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Resolve which class from the metadata the section should bind to.

    The LLM may provide a ``className`` or ``classId`` in the section spec.
    Falls back to the first class in metadata if nothing matches.
    """
    if not class_metadata:
        return None
    class_name = _clean_text(section_spec.get("className"))
    class_id = _clean_text(section_spec.get("classId"))

    # Try matching by ID first
    if class_id:
        for cls in class_metadata:
            if cls["id"] == class_id:
                return cls
    # Try matching by name (case-insensitive)
    if class_name:
        for cls in class_metadata:
            if cls["name"].lower() == class_name.lower():
                return cls
    # Fallback: first class with attributes
    for cls in class_metadata:
        if cls.get("attributes"):
            return cls
    return class_metadata[0] if class_metadata else None


def _pick_label_field(cls: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the best string attribute for chart label-field.

    Prefers meaningful string attributes (skipping 'id') so that chart
    labels show human-readable values like 'Nike' instead of 'S001'.
    """
    attrs = cls.get("attributes", [])
    # First pass: skip attributes named 'id'
    for a in attrs:
        if a.get("isString") and a.get("name", "").lower() != "id":
            return a
    # Second pass: accept 'id' if it's the only string attribute
    for a in attrs:
        if a.get("isString"):
            return a
    return attrs[0] if attrs else None


def _pick_data_field(cls: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the best numeric attribute for chart data-field."""
    attrs = cls.get("attributes", [])
    for a in attrs:
        if a.get("isNumeric"):
            return a
    return attrs[0] if attrs else None


def _dummy_chart_data(chart_type: str) -> List[Dict[str, Any]]:
    """Return sample preview data appropriate for *chart_type*.

    This data is shown inside the GrapesJS editor so that LLM-generated
    charts are not empty placeholders.  At runtime the data is replaced
    by real values fetched from the data source.
    """
    if chart_type == "radar-chart":
        return [
            {"subject": "Category A", "value": 85, "fullMark": 100},
            {"subject": "Category B", "value": 75, "fullMark": 100},
            {"subject": "Category C", "value": 90, "fullMark": 100},
            {"subject": "Category D", "value": 80, "fullMark": 100},
            {"subject": "Category E", "value": 70, "fullMark": 100},
        ]
    # bar-chart, line-chart, and general fallback
    return [
        {"name": "Category A", "value": 40},
        {"name": "Category B", "value": 65},
        {"name": "Category C", "value": 85},
        {"name": "Category D", "value": 55},
        {"name": "Category E", "value": 75},
    ]


def _dummy_pie_data() -> List[Dict[str, Any]]:
    """Return sample preview data for pie / radial-bar charts."""
    return [
        {"name": "Desktop", "value": 45, "color": "#0088FE"},
        {"name": "Mobile", "value": 35, "color": "#00C49F"},
        {"name": "Tablet", "value": 15, "color": "#FFBB28"},
        {"name": "Other", "value": 5, "color": "#FF8042"},
    ]


def _convert_table_rows_to_chart_data(
    rows: List[Dict[str, Any]],
    chart_type: str,
    cls: Optional[Dict[str, Any]] = None,
    value_attr_name: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Convert table-format rows to chart-format data points.

    When the LLM provides ``sampleData`` in table format (keyed by column
    names, e.g. ``{"brand": "Nike", "size": 42}``), this function picks
    the best string column as label and the best numeric column as value
    to produce ``{"name": "Nike", "value": 42}``.

    If *value_attr_name* is given, that specific column is used as the
    value source instead of auto-detecting the first numeric column.
    This allows extracting per-series data (e.g. "size" for the Size
    series and "price" for the Price series).
    """
    if not rows:
        return None

    first = rows[0]

    # --- Determine label key (first string-valued column) ---
    label_key: Optional[str] = None
    # If we have class metadata, prefer the label-field attribute
    if cls:
        lf = _pick_label_field(cls)
        if lf and lf["name"] in first:
            label_key = lf["name"]
    if not label_key:
        for k, v in first.items():
            if isinstance(v, str) and k.lower() not in ("id", "imageurl", "image_url", "url", "description"):
                label_key = k
                break
    if not label_key:
        # Fallback: any string at all
        for k, v in first.items():
            if isinstance(v, str):
                label_key = k
                break

    # --- Determine value key ---
    value_key: Optional[str] = None
    # If a specific attribute name was requested, find it (case-insensitive)
    if value_attr_name:
        for k in first:
            if k.lower() == value_attr_name.lower():
                value_key = k
                break
    # Auto-detect from class metadata
    if not value_key and cls:
        df = _pick_data_field(cls)
        if df and df["name"] in first:
            value_key = df["name"]
    # Auto-detect first numeric column
    if not value_key:
        for k, v in first.items():
            if isinstance(v, (int, float)) and k != label_key:
                value_key = k
                break

    if not label_key or not value_key:
        return None

    converted: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        label = row.get(label_key, f"Item {idx + 1}")
        value = row.get(value_key, 0)
        if chart_type == "radar-chart":
            converted.append({"subject": str(label), "value": value, "fullMark": 100})
        elif chart_type == "pie-chart":
            converted.append({"name": str(label), "value": value,
                              "color": _PIE_COLORS[idx % len(_PIE_COLORS)]})
        else:
            converted.append({"name": str(label), "value": value})
    return converted if converted else None


def _extract_sample_data(
    section_spec: Dict[str, Any],
    chart_type: str,
    cls: Optional[Dict[str, Any]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Extract LLM-provided sample data from the section spec.

    The LLM is instructed to include a ``sampleData`` array with
    realistic preview rows.  Returns ``None`` if nothing usable was
    provided, so callers can fall back to generic dummy data.

    Handles two formats:
    1. **Chart format** — ``{"name": "...", "value": 42}`` (used directly)
    2. **Table format** — ``{"brand": "Nike", "size": 42, ...}`` (converted
       automatically by picking the best label/numeric columns)
    """
    raw = section_spec.get("sampleData")
    if not isinstance(raw, list) or not raw:
        return None

    # --- Try chart-native format first ---
    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if chart_type == "radar-chart":
            if "subject" in item and "value" in item:
                cleaned.append({**item, "value": _coerce_numeric(item.get("value"))})
            elif "name" in item and "value" in item:
                cleaned.append({"subject": item["name"],
                                "value": _coerce_numeric(item.get("value")),
                                "fullMark": item.get("fullMark", 100)})
        elif chart_type == "pie-chart":
            if "name" in item and "value" in item:
                cleaned.append({**item, "value": _coerce_numeric(item.get("value"))})
        else:
            if "name" in item and "value" in item:
                cleaned.append({**item, "value": _coerce_numeric(item.get("value"))})
    if cleaned:
        return cleaned

    # --- Fallback: convert table-format rows to chart format ---
    table_rows = [item for item in raw if isinstance(item, dict)]
    return _convert_table_rows_to_chart_data(table_rows, chart_type, cls)


def _build_series(
    chart_type: str,
    cls: Dict[str, Any],
    section_spec: Dict[str, Any],
) -> str:
    """Build the JSON-serialized series array for a chart component."""
    label_attr = _pick_label_field(cls)
    data_attr = _pick_data_field(cls)

    # Try to use LLM-provided contextual sample data, else generic fallback
    llm_data = _extract_sample_data(section_spec, chart_type, cls)
    fallback_data = llm_data or _dummy_chart_data(chart_type)

    series_list: List[Dict[str, Any]] = []
    # If the LLM provided explicit series, use them
    raw_series = section_spec.get("series")
    if isinstance(raw_series, list) and raw_series:
        for idx, raw in enumerate(raw_series):
            if not isinstance(raw, dict):
                continue
            # Per-series sample data takes priority over section-level
            per_series_data = None
            raw_data = raw.get("data")
            if isinstance(raw_data, list) and raw_data and all(isinstance(d, dict) for d in raw_data):
                per_series_data = raw_data
            s: Dict[str, Any] = {
                "name": _clean_text(raw.get("name"), fallback=f"Series {idx + 1}"),
                "data-source": raw.get("classId") or cls["id"],
                "color": raw.get("color") or _CHART_COLORS[idx % len(_CHART_COLORS)],
                "data": per_series_data or fallback_data,
            }
            # Resolve label/data fields
            lf = raw.get("labelField") or raw.get("label-field")
            df = raw.get("dataField") or raw.get("data-field")
            if lf:
                s["label-field"] = lf
            elif label_attr:
                s["label-field"] = label_attr["id"]
            if df:
                s["data-field"] = df
            elif data_attr:
                s["data-field"] = data_attr["id"]
            series_list.append(s)
    else:
        # Auto-generate one series per numeric attribute (up to 3).
        # Each series gets its OWN data extracted from the LLM sample rows
        # so that e.g. the "Size" series shows size values and the "Price"
        # series shows price values (instead of all sharing one column).
        numeric_attrs = [a for a in cls.get("attributes", []) if a.get("isNumeric")]
        if not numeric_attrs:
            numeric_attrs = cls.get("attributes", [])[:1]

        # Grab raw table-format sampleData once for per-attribute extraction
        raw_sample = section_spec.get("sampleData")
        table_rows: Optional[List[Dict[str, Any]]] = None
        if isinstance(raw_sample, list) and raw_sample:
            candidate = [r for r in raw_sample if isinstance(r, dict)]
            # Only treat as table rows if NOT already in chart-native format
            if candidate and "name" not in candidate[0] and "subject" not in candidate[0]:
                table_rows = candidate

        for idx, num_attr in enumerate(numeric_attrs[:3]):
            # Try per-attribute extraction from table rows
            per_attr_data = None
            if table_rows:
                per_attr_data = _convert_table_rows_to_chart_data(
                    table_rows, chart_type, cls, value_attr_name=num_attr["name"],
                )
            s = {
                "name": num_attr["name"].replace("_", " ").title(),
                "data-source": cls["id"],
                "color": _CHART_COLORS[idx % len(_CHART_COLORS)],
                "data": per_attr_data or fallback_data,
            }
            if label_attr:
                s["label-field"] = label_attr["id"]
            s["data-field"] = num_attr["id"]
            series_list.append(s)

    if not series_list:
        series_list = [{
            "name": "Series 1",
            "data-source": cls["id"],
            "color": _CHART_COLORS[0],
            "data": fallback_data,
        }]

    return json.dumps(series_list)


def _chart_component(
    chart_type: str,
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a GrapesJS chart component (bar-chart, pie-chart, line-chart, etc.)
    bound to class diagram data.
    """
    title = _clean_text(section_spec.get("title"), fallback=chart_type.replace("-", " ").title())
    cls = _resolve_class_binding(section_spec, class_metadata)

    # Chart attributes that GrapesJS chart components expect
    chart_attrs: Dict[str, Any] = {
        "class": f"{chart_type}-component",
        "chart-title": title,
        "show-grid": "true",
        "show-legend": "true",
    }

    if cls:
        label_attr = _pick_label_field(cls)
        data_attr = _pick_data_field(cls)

        # For pie-chart: data-source, label-field, data-field go directly on attrs
        if chart_type == "pie-chart":
            chart_attrs["data-source"] = cls["id"]
            if label_attr:
                chart_attrs["label-field"] = label_attr["id"]
            if data_attr:
                chart_attrs["data-field"] = data_attr["id"]
            # Use LLM-provided sample data for pie if available
            llm_pie = _extract_sample_data(section_spec, "pie-chart", cls)
            chart_attrs["series"] = json.dumps([{
                "name": cls["name"],
                "data-source": cls["id"],
                "color": _PIE_COLORS[0],
                "data": llm_pie or _dummy_pie_data(),
            }])
        else:
            # For line/bar/radar charts: binding goes inside the series
            chart_attrs["series"] = _build_series(chart_type, cls, section_spec)
    else:
        # No class binding — use LLM sample data or generic fallback
        llm_data = _extract_sample_data(section_spec, chart_type, None)
        if chart_type == "pie-chart":
            chart_attrs["series"] = json.dumps([{
                "name": "Series 1",
                "color": _PIE_COLORS[0],
                "data": llm_data or _dummy_pie_data(),
            }])
        else:
            chart_attrs["series"] = json.dumps([{
                "name": "Series 1",
                "color": _CHART_COLORS[0],
                "data": llm_data or _dummy_chart_data(chart_type),
            }])

    # Chart-type specific defaults
    if chart_type == "bar-chart":
        chart_attrs.setdefault("bar-width", "30")
        chart_attrs.setdefault("orientation", "vertical")
        chart_attrs.setdefault("stacked", "false")
    elif chart_type == "line-chart":
        chart_attrs.setdefault("line-width", "2")
        chart_attrs.setdefault("curve-type", "monotone")
        chart_attrs.setdefault("show-tooltip", "true")
        chart_attrs.setdefault("animate", "true")
    elif chart_type == "pie-chart":
        chart_attrs.setdefault("legend-position", "bottom")
        chart_attrs.setdefault("show-labels", "true")
        chart_attrs.setdefault("label-position", "inside")
        chart_attrs.setdefault("padding-angle", "0")
    elif chart_type == "radar-chart":
        chart_attrs.setdefault("show-tooltip", "true")
        chart_attrs.setdefault("show-radius-axis", "true")

    return {
        "type": chart_type,
        "attributes": chart_attrs,
        "style": {
            "width": "100%",
            "min-height": "400px",
            "margin": "12px 0",
            "border-radius": "12px",
        },
    }


def _coerce_table_rows(
    src: Dict[str, Any], columns: List[str], sample: List[Any]
) -> "tuple[List[str], List[List[str]]]":
    """Best-effort REAL rows for a standalone (class-less) data table.

    Priority: explicit ``rows`` (each a ``{cells:[...]}`` aligned to columns) →
    chart-style ``sampleData`` (name/value → 2 columns) → a couple of synthesized
    rows from the column names. NEVER returns an empty body — the whole point is
    that a class-less table stops rendering as one blank cell.
    """
    rows: List[List[str]] = []
    for r in (src.get("rows") or []):
        cells = r.get("cells") if isinstance(r, dict) else (r if isinstance(r, list) else None)
        if isinstance(cells, list):
            vals = [_clean_text(str(c)) for c in cells if c is not None]
            if any(vals):
                rows.append(vals)
    if rows:
        return columns, rows

    sv: List[List[str]] = []
    for d in (sample or []):
        if isinstance(d, dict) and _clean_text(str(d.get("name", ""))):
            val = d.get("value")
            sv.append([_clean_text(str(d.get("name"))), "" if val is None else _clean_text(str(val))])
    if sv:
        cols = columns if len(columns) == 2 else ["Item", "Value"]
        return cols, sv

    if columns:
        # A few neutral rows so the table has structure even with no data.
        rows = [["—" for _ in columns] for _ in range(3)]
    return columns, rows


def _html_data_table_node(columns: List[str], rows: List[List[str]]) -> Dict[str, Any]:
    """A themed, ALWAYS-populated ``<table class='ds-table'>`` component node.

    Used for standalone GUIs (no ClassDiagram) where the data-bound table widget
    would render empty — a plain HTML table with real rows renders directly.
    """
    cols = [str(c) for c in (columns or []) if _clean_text(str(c))] or ["Item", "Value"]
    head = "".join(f"<th>{_html_escape(str(c))}</th>" for c in cols)
    body = ""
    for r in (rows or []):
        cells = [str(x) for x in r][: len(cols)]
        cells += [""] * (len(cols) - len(cells))
        body += "<tr>" + "".join(f"<td>{_html_escape(c)}</td>" for c in cells) + "</tr>"
    markup = f"<table class='ds-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    nodes = html_to_components(markup)
    return nodes[0] if nodes else {"tagName": "table", "attributes": {"class": "ds-table"}}


def _table_component(
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a data table for a section.

    With a resolved ClassDiagram binding → the recognizer-compatible data-bound
    ``type:'table'`` widget (columns from the class, rows from its data-source).
    WITHOUT a class (the common standalone-GUI case) → a themed HTML table
    populated from the LLM's ``fields`` + ``rows``/``sampleData`` so it renders
    with real content instead of a single empty cell.
    """
    title = _clean_text(section_spec.get("title"), fallback="Data Table")
    cls = _resolve_class_binding(section_spec, class_metadata)

    if not cls:
        columns = [c for c in (section_spec.get("fields") or []) if isinstance(c, str) and c.strip()]
        cols, rows = _coerce_table_rows(section_spec, columns, section_spec.get("sampleData") or [])
        return _html_data_table_node(cols, rows)

    table_attrs: Dict[str, Any] = {
        "class": "table-component",
        "chart-title": title,
        "show-header": "true",
        "striped-rows": "false",
        "show-pagination": "true",
        "action-buttons": "true",
        "rows-per-page": "5",
    }

    if cls:
        table_attrs["data-source"] = cls["id"]
        # Build auto-generated columns (field columns + lookup columns)
        auto_columns: List[Dict[str, Any]] = []

        # Field columns from attributes
        for attr in cls.get("attributes", []):
            auto_columns.append({
                "field": attr["name"],
                "label": attr["name"].replace("_", " ").title(),
                "columnType": "field",
                "_expanded": False,
            })

        # Lookup columns from association ends
        for end in cls.get("associationEnds", []):
            auto_columns.append({
                "field": end.get("targetClassName", end.get("targetClassId", "")),
                "label": end.get("targetClassName", "Related").replace("_", " ").title(),
                "columnType": "lookup",
                "lookupEntity": end.get("targetClassId", ""),
                "lookupField": end.get("displayAttributeName", ""),
                "_expanded": False,
            })

        if auto_columns:
            table_attrs["columns"] = json.dumps(auto_columns)

    return {
        "type": "table",
        "attributes": table_attrs,
        "style": {
            "width": "100%",
            "min-height": "300px",
            "margin": "12px 0",
            "border-radius": "12px",
        },
    }


def _dashboard_component(
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a dashboard section with a table + charts for the bound class."""
    title = _clean_text(section_spec.get("title"), fallback="Dashboard")
    cls = _resolve_class_binding(section_spec, class_metadata)

    # Build sub-components: a table + up to 2 charts
    components: List[Dict[str, Any]] = [
        {
            "tagName": "h2",
            "content": title,
            "style": {
                "margin": "0 0 20px 0",
                "font-size": "1.35rem",
                "font-weight": "700",
                "color": "#0f172a",
            },
        },
    ]

    # Table
    table_spec = dict(section_spec)
    table_spec["type"] = "table"
    table_spec["title"] = f"{cls['name']} Data" if cls else "Data Table"
    components.append(_table_component(table_spec, class_metadata))

    if cls:
        numeric_attrs = [a for a in cls.get("attributes", []) if a.get("isNumeric")]
        if numeric_attrs:
            # Bar chart
            chart_spec = dict(section_spec)
            chart_spec["type"] = "bar_chart"
            chart_spec["title"] = f"{cls['name']} Overview"
            components.append(_chart_component("bar-chart", chart_spec, class_metadata))

            if len(numeric_attrs) >= 2:
                # Pie chart for the second numeric attribute
                pie_spec = dict(section_spec)
                pie_spec["type"] = "pie_chart"
                pie_spec["title"] = f"{cls['name']} Distribution"
                components.append(_chart_component("pie-chart", pie_spec, class_metadata))

    # Charts grid container
    chart_grid: Dict[str, Any] = {
        "tagName": "div",
        "style": {
            "display": "grid",
            "grid-template-columns": "1fr 1fr",
            "gap": "20px",
            "margin-top": "20px",
        },
        "components": components[2:],  # Charts only (skip h2 + table)
    }

    return {
        "tagName": "section",
        "attributes": {"class": "assistant-dashboard"},
        "style": {
            "padding": "32px",
            "background-color": "#ffffff",
            "border-radius": "14px",
            "margin": "12px 24px",
            "box-shadow": "0 1px 4px rgba(0,0,0,0.06)",
            "border": "1px solid #f1f5f9",
        },
        "components": [components[0], components[1], chart_grid] if len(components) > 2 else components,
    }


# ---------------------------------------------------------------------------
# Card wrapper — wraps bare data components (table, chart) inside a styled card
# ---------------------------------------------------------------------------

def _card_wrap(title: str, inner: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a bare data component (table, chart) inside a styled card with a title."""
    components: List[Dict[str, Any]] = []
    if title:
        components.append({
            "tagName": "h2",
            "content": title,
            "style": {
                "margin": "0 0 16px 0",
                "font-size": "1.25rem",
                "font-weight": "700",
                "color": "#0f172a",
            },
        })
    components.append(inner)
    return {
        "tagName": "section",
        "attributes": {"class": "assistant-card"},
        "style": {
            "padding": "28px",
            "background-color": "#ffffff",
            "border-radius": "14px",
            "margin": "12px 24px",
            "box-shadow": "0 1px 4px rgba(0,0,0,0.06)",
            "border": "1px solid #f1f5f9",
        },
        "components": components,
    }


# ---------------------------------------------------------------------------
# Metric card component
# ---------------------------------------------------------------------------

def _metric_card_component(
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a GrapesJS metric-card component bound to a class attribute."""
    title = _clean_text(section_spec.get("title"), fallback="Metric")
    cls = _resolve_class_binding(section_spec, class_metadata)

    card_attrs: Dict[str, Any] = {
        "class": "metric-card-component",
        "metric-title": title,
        "value-color": "#2c3e50",
        "value-size": "32",
        "show-trend": "true",
        "positive-color": "#27ae60",
        "negative-color": "#e74c3c",
        "format": "number",
    }

    if cls:
        card_attrs["data-source"] = cls["id"]
        # Pick the best numeric attribute for the metric
        data_attr = _pick_data_field(cls)
        if data_attr:
            card_attrs["data-field"] = data_attr["id"]

    return {
        "type": "metric-card",
        "attributes": card_attrs,
        "style": {
            "width": "100%",
            "min-height": "140px",
            "margin": "8px 0",
        },
    }


# ---------------------------------------------------------------------------
# Stats grid — multiple metric cards in a responsive grid
# ---------------------------------------------------------------------------

def _stats_grid_component(
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a grid of metric cards from a list of stat items.

    Reads the typed ``stats`` list (``{label, value}`` pairs) first, then
    falls back to ``items`` — which on the free-text complete-system path
    arrives as ``{label, value}`` dicts (or plain label strings). The
    LLM-provided ``value`` is preserved on the card (issue #7: figures used
    to be silently discarded). For the class-bound case each card binds to a
    DISTINCT numeric attribute instead of all cards sharing the first field.
    """
    title = _clean_text(section_spec.get("title"), fallback="Key Metrics")
    raw_stats = section_spec.get("stats") if isinstance(section_spec.get("stats"), list) else []
    if not raw_stats:
        raw_stats = section_spec.get("items") if isinstance(section_spec.get("items"), list) else []

    # Resolve class metadata for data binding
    cls = _resolve_class_binding(section_spec, class_metadata)

    # Distinct numeric fields so each card binds to its OWN attribute rather
    # than every card showing the same first numeric field.
    numeric_attrs: List[Dict[str, Any]] = []
    if cls:
        numeric_attrs = [a for a in cls.get("attributes", []) if a.get("isNumeric")]
        if not numeric_attrs:
            fallback_attr = _pick_data_field(cls)
            numeric_attrs = [fallback_attr] if fallback_attr else []

    cards: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_stats):
        if isinstance(item, dict):
            label = _clean_text(item.get("label") or item.get("name"), fallback="Metric")
            value = _clean_text(item.get("value"))
            fmt = _clean_text(item.get("format"), fallback="number")
        elif isinstance(item, str):
            label = item
            value = ""
            fmt = "number"
        else:
            continue

        card_attrs: Dict[str, Any] = {
            "class": "metric-card-component",
            "metric-title": label,
            "value-color": "#2c3e50",
            "value-size": "32",
            "show-trend": "true",
            "positive-color": "#27ae60",
            "negative-color": "#e74c3c",
            "format": fmt,
        }
        # Preserve the LLM-provided figure so it renders instead of being
        # discarded and recomputed to nothing.
        if value:
            card_attrs["metric-value"] = value
        if cls:
            card_attrs["data-source"] = cls["id"]
            if numeric_attrs:
                data_attr = numeric_attrs[idx % len(numeric_attrs)]
                if data_attr:
                    card_attrs["data-field"] = data_attr["id"]

        cards.append({
            "type": "metric-card",
            "attributes": card_attrs,
            "style": {
                "width": "100%",
                "min-height": "140px",
            },
        })

    if not cards:
        cards = [
            _stat_placeholder("Total", cls),
            _stat_placeholder("Active", cls),
            _stat_placeholder("Growth", cls),
        ]

    col_count = min(len(cards), 4)
    return {
        "tagName": "section",
        "attributes": {"class": "assistant-stats-grid"},
        "style": {
            "display": "grid",
            "grid-template-columns": f"repeat({col_count}, 1fr)",
            "gap": "16px",
            "margin": "12px 24px",
        },
        "components": [
            *([
                {
                    "tagName": "h2",
                    "content": title,
                    "style": {
                        "grid-column": f"1 / -1",
                        "font-size": "1.25rem",
                        "font-weight": "700",
                        "color": "#0f172a",
                        "margin": "0 0 4px 0",
                    },
                }
            ] if title else []),
            *cards,
        ],
    }


def _stat_placeholder(label: str, cls: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a real metric-card component as a placeholder stat card."""
    card_attrs: Dict[str, Any] = {
        "class": "metric-card-component",
        "metric-title": label,
        "value-color": "#2c3e50",
        "value-size": "32",
        "show-trend": "true",
        "positive-color": "#27ae60",
        "negative-color": "#e74c3c",
        "format": "number",
    }
    if cls:
        card_attrs["data-source"] = cls["id"]
        data_attr = _pick_data_field(cls)
        if data_attr:
            card_attrs["data-field"] = data_attr["id"]
    return {
        "type": "metric-card",
        "attributes": card_attrs,
        "style": {
            "width": "100%",
            "min-height": "140px",
        },
    }


# ---------------------------------------------------------------------------
# Footer component
# ---------------------------------------------------------------------------

def _footer_component(title: str, body: str, items: List[str]) -> Dict[str, Any]:
    """Build a page footer with project name and optional link labels."""
    link_components = []
    for item in items:
        if isinstance(item, str) and item.strip():
            link_components.append({
                "tagName": "a",
                "attributes": {"href": "#"},
                "content": item.strip(),
                "style": {
                    "color": "#94a3b8",
                    "text-decoration": "none",
                    "font-size": "0.85rem",
                    "transition": "color 0.2s",
                },
            })

    return {
        "tagName": "footer",
        "attributes": {"class": "assistant-footer"},
        "style": {
            "padding": "32px 48px",
            "background-color": "#0f172a",
            "color": "#94a3b8",
            "display": "flex",
            "justify-content": "space-between",
            "align-items": "center",
            "margin-top": "24px",
            "font-family": "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
        },
        "components": [
            {
                "tagName": "div",
                "components": [
                    {
                        "tagName": "div",
                        "content": title or "Project",
                        "style": {
                            "font-weight": "700",
                            "font-size": "1.1rem",
                            "color": "#ffffff",
                            "margin-bottom": "4px",
                        },
                    },
                    {
                        "tagName": "div",
                        "content": body or "\u00a9 2026 All rights reserved.",
                        "style": {"font-size": "0.8rem"},
                    },
                ],
            },
            *([
                {
                    "tagName": "div",
                    "style": {"display": "flex", "gap": "20px"},
                    "components": link_components,
                }
            ] if link_components else []),
        ],
    }


# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------

def _two_column_component(
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a two-column layout.  The LLM provides left/right sub-sections."""
    title = _clean_text(section_spec.get("title"), fallback="")
    left_spec = section_spec.get("left") if isinstance(section_spec.get("left"), dict) else {}
    right_spec = section_spec.get("right") if isinstance(section_spec.get("right"), dict) else {}

    left_child = _build_section_component(left_spec, class_metadata) if left_spec else _content_component("Left", "Content")
    right_child = _build_section_component(right_spec, class_metadata) if right_spec else _content_component("Right", "Content")

    # Remove outer margins from children since the grid handles spacing
    for child in (left_child, right_child):
        if isinstance(child, dict) and "style" in child:
            child["style"]["margin"] = "0"

    components: List[Dict[str, Any]] = []
    if title:
        components.append({
            "tagName": "h2",
            "content": title,
            "style": {
                "grid-column": "1 / -1",
                "font-size": "1.35rem",
                "font-weight": "700",
                "color": "#0f172a",
                "margin": "0 0 8px 0",
            },
        })
    components.extend([left_child, right_child])

    return {
        "tagName": "section",
        "attributes": {"class": "assistant-two-column"},
        "style": {
            "display": "grid",
            "grid-template-columns": "1fr 1fr",
            "gap": "20px",
            "margin": "12px 24px",
        },
        "components": components,
    }


# ---------------------------------------------------------------------------
# Action button component (method execution)
# ---------------------------------------------------------------------------

def _action_button_component(
    method: Dict[str, Any],
    cls: Dict[str, Any],
    table_id: str = "",
) -> Dict[str, Any]:
    """Build a GrapesJS action-button component for a class method."""
    return {
        "type": "action-button",
        "content": method["name"],
        "attributes": {
            "class": "action-button-component",
            "type": "button",
            "data-button-label": method["name"],
            "data-action-type": "run-method",
            "data-method-class": cls["id"],
            "data-method": method["id"],
            "data-instance-source": table_id,
            "instance-method": "true" if method.get("isInstanceMethod") else "false",
        },
        "button-label": method["name"],
        "action-type": "run-method",
        "method-class": cls["id"],
        "method": method["id"],
        "instance-source": table_id,
        "confirmation-required": False,
        "style": {
            "display": "inline-flex",
            "align-items": "center",
            "padding": "6px 14px",
            "background": "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)",
            "color": "#fff",
            "border-radius": "4px",
            "font-size": "13px",
            "font-weight": "600",
            "cursor": "pointer",
            "border": "none",
            "margin": "4px",
        },
    }


def _method_buttons_row(cls: Dict[str, Any], table_id: str = "") -> Optional[Dict[str, Any]]:
    """Build a row of action buttons for all methods in *cls*.

    Returns ``None`` when the class has no methods.
    """
    methods = cls.get("methods", [])
    if not methods:
        return None

    buttons = [_action_button_component(m, cls, table_id) for m in methods]
    return {
        "tagName": "div",
        "attributes": {"class": "assistant-method-buttons"},
        "style": {
            "display": "flex",
            "flex-wrap": "wrap",
            "gap": "8px",
            "margin": "16px 0",
        },
        "components": buttons,
    }


# ---------------------------------------------------------------------------
# Navigation sidebar
# ---------------------------------------------------------------------------

def _nav_header_component(
    page_names: List[str],
    active_page: str = "",
    project_name: str = "BESSER",
    tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a horizontal navigation header bar with links to all pages.

    This is injected at the top of every LLM-generated page so users can
    navigate between pages.  The active page link is visually highlighted.
    Uses a clean white design with subtle accents.
    """
    # Theme-aware: the nav is part of the app, so it wears the app's tokens —
    # the old hardcoded blue/Inter header visibly clashed with themed pages.
    t = tokens or theme_tokens("default")
    palette = t.get("palette", {})
    primary = palette.get("primary", "#2563eb")
    try:
        active_bg = _hex_to_rgba(primary, 0.10)
    except Exception:
        active_bg = "transparent"

    nav_links: List[Dict[str, Any]] = []
    for name in page_names:
        route = f"/{re.sub(r'[^a-z0-9-]+', '-', name.lower()).strip('-') or 'page'}"
        is_active = name.lower() == active_page.lower()
        nav_links.append({
            "type": "link",
            "attributes": {
                "href": route,
                "data-navigate-to": name.lower().replace(" ", "-"),
            },
            "style": {
                "color": primary if is_active else palette.get("muted", "#64748b"),
                "text-decoration": "none",
                "font-weight": "600" if is_active else "500",
                "font-size": "0.9rem",
                "padding": "8px 16px",
                "border-radius": t.get("radius", "8px"),
                "background-color": active_bg if is_active else "transparent",
                "transition": "all 0.2s",
            },
            "components": [{"type": "textnode", "content": name}],
        })

    return {
        "tagName": "nav",
        "attributes": {"class": "assistant-nav-header"},
        "style": {
            "background-color": palette.get("surface", "#ffffff"),
            "padding": "0 32px",
            "height": "64px",
            "display": "flex",
            "justify-content": "space-between",
            "align-items": "center",
            "font-family": t.get("font", "'Segoe UI', system-ui, sans-serif"),
            "box-shadow": "0 1px 3px rgba(0,0,0,0.06)",
            "border-bottom": f"1px solid {palette.get('border', '#e2e8f0')}",
            "position": "sticky",
            "top": "0",
            "z-index": "50",
        },
        "components": [
            {
                "type": "text",
                "style": {
                    "font-size": "1.25rem",
                    "font-weight": "700",
                    "color": palette.get("text", "#0f172a"),
                    "letter-spacing": "-0.01em",
                },
                "components": [{"type": "textnode", "content": project_name}],
            },
            {
                "style": {
                    "display": "flex",
                    "gap": "4px",
                    "align-items": "center",
                },
                "components": nav_links,
            },
        ],
    }


def _nav_sidebar_component(class_metadata: List[Dict[str, Any]], active_index: int = 0) -> Dict[str, Any]:
    """Build a navigation sidebar with links to all class pages."""
    nav_links: List[Dict[str, Any]] = []
    for idx, cls in enumerate(class_metadata):
        page_name = cls["name"].lower().replace(" ", "-")
        is_active = idx == active_index
        nav_links.append({
            "tagName": "a",
            "content": cls["name"],
            "attributes": {
                "href": f"/{page_name}",
                "data-navigate-to": page_name,
                "class": "nav-link" + (" active" if is_active else ""),
            },
            "style": {
                "display": "block",
                "padding": "10px 20px",
                "color": "#e2e8f0" if not is_active else "#ffffff",
                "text-decoration": "none",
                "font-weight": "600" if is_active else "400",
                "font-size": "0.9rem",
                "border-left": "3px solid " + ("#38bdf8" if is_active else "transparent"),
                "background-color": "rgba(255,255,255,0.1)" if is_active else "transparent",
                "transition": "all 0.2s",
            },
        })

    return {
        "tagName": "nav",
        "attributes": {"class": "assistant-nav-sidebar"},
        "style": {
            "width": "250px",
            "min-height": "100vh",
            "background": "linear-gradient(180deg, #4b3c82 0%, #5a3d91 100%)",
            "padding": "20px 0",
            "flex-shrink": "0",
        },
        "components": [
            {
                "tagName": "div",
                "style": {"padding": "0 20px 20px 20px", "border-bottom": "1px solid rgba(255,255,255,0.15)"},
                "components": [
                    {
                        "tagName": "h2",
                        "content": "BESSER",
                        "style": {"color": "#ffffff", "font-size": "1.4rem", "margin": "0", "font-weight": "700"},
                    },
                ],
            },
            {
                "tagName": "div",
                "style": {"padding-top": "16px"},
                "components": nav_links,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Full class page builder  (mirrors frontend autoGenerateGUIFromClassDiagram)
# ---------------------------------------------------------------------------

def _build_class_page(
    cls: Dict[str, Any],
    class_metadata: List[Dict[str, Any]],
    page_counter: int,
) -> Dict[str, Any]:
    """Build one GrapesJS page for a single class, matching the auto-generate layout.

    Layout:
    ┌───────────┬──────────────────────────────────────────────┐
    │ Nav       │ Page Title                                   │
    │ sidebar   │ Description                                  │
    │           │ [Data Table – bound to class]                │
    │           │ [Method buttons – for each method]           │
    │           │ [Bar Chart – if numeric attrs]               │
    └───────────┴──────────────────────────────────────────────┘
    """
    class_name = cls["name"]
    page_name = class_name.lower().replace(" ", "-")
    table_id = f"table-{page_name}-{page_counter}"

    # -- Sidebar --
    sidebar = _nav_sidebar_component(class_metadata, active_index=page_counter)

    # -- Main content components --
    main_children: List[Dict[str, Any]] = [
        # Page title
        {
            "tagName": "h1",
            "content": class_name,
            "style": {
                "margin": "0 0 8px 0",
                "font-size": "1.75rem",
                "font-weight": "700",
                "color": "#1e293b",
            },
        },
        # Description
        {
            "tagName": "p",
            "content": f"Manage {class_name} data",
            "style": {"margin": "0 0 24px 0", "color": "#64748b", "font-size": "0.95rem"},
        },
    ]

    # -- Data table bound to class --
    table_spec: Dict[str, Any] = {"className": class_name, "title": f"{class_name} List"}
    table_comp = _table_component(table_spec, class_metadata)
    # Inject a stable ID for button linkage
    table_comp.setdefault("attributes", {})["id"] = table_id
    main_children.append(table_comp)

    # -- Method buttons --
    buttons_row = _method_buttons_row(cls, table_id=table_id)
    if buttons_row:
        main_children.append(buttons_row)

    # -- Charts (bar chart if numeric attrs exist) --
    numeric_attrs = [a for a in cls.get("attributes", []) if a.get("isNumeric")]
    if numeric_attrs:
        chart_spec: Dict[str, Any] = {"className": class_name, "title": f"{class_name} Overview"}
        main_children.append(_chart_component("bar-chart", chart_spec, class_metadata))

    # -- Main content area --
    main_area: Dict[str, Any] = {
        "tagName": "main",
        "style": {
            "flex": "1",
            "padding": "32px",
            "background-color": "#f1f5f9",
            "min-height": "100vh",
            "overflow-y": "auto",
        },
        "components": main_children,
    }

    # -- Root layout (flex row) --
    root: Dict[str, Any] = {
        "tagName": "div",
        "attributes": {"class": "assistant-page-layout"},
        "style": {"display": "flex", "min-height": "100vh"},
        "components": [sidebar, main_area],
    }

    wrapper = _default_wrapper_component()
    wrapper["components"] = [root]

    return {
        "name": class_name,
        "route_path": f"/{page_name}",
        "frames": [{"component": wrapper}],
    }




def _legacy_section_component(section_spec: Dict[str, Any], class_metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    section_type = _clean_text(section_spec.get("type"), fallback="content").lower()
    title = _clean_text(section_spec.get("title"), fallback="New Section")
    body = _clean_text(section_spec.get("body"), fallback="Section content")
    cta_label = _clean_text(section_spec.get("ctaLabel"), fallback="Continue")
    items = section_spec.get("items") if isinstance(section_spec.get("items"), list) else []
    fields = section_spec.get("fields") if isinstance(section_spec.get("fields"), list) else []

    if section_type in {"hero", "landing"}:
        return _hero_component(title, body, cta_label)
    if section_type in {"feature_list", "features", "list"}:
        return _feature_list_component(title, [str(item) for item in items])
    if section_type in {"form", "contact_form", "signup_form"}:
        return _form_component(title, [str(field) for field in fields], cta_label)
    if section_type in {"footer"}:
        return _footer_component(title, body, [str(i) for i in items])
    if section_type in {"stats_grid", "stats-grid", "stats", "metrics_grid", "metrics"}:
        return _stats_grid_component(section_spec, class_metadata)
    if section_type in {"two_column", "two-column", "split", "columns"}:
        return _two_column_component(section_spec, class_metadata)

    # ── Data-bound components (charts, tables, dashboards) ──────────
    if section_type in {"table", "data_table"}:
        return _card_wrap(title, _table_component(section_spec, class_metadata))
    if section_type in {"bar_chart", "bar-chart", "barchart"}:
        return _card_wrap(title, _chart_component("bar-chart", section_spec, class_metadata))
    if section_type in {"pie_chart", "pie-chart", "piechart"}:
        return _card_wrap(title, _chart_component("pie-chart", section_spec, class_metadata))
    if section_type in {"line_chart", "line-chart", "linechart"}:
        return _card_wrap(title, _chart_component("line-chart", section_spec, class_metadata))
    if section_type in {"radar_chart", "radar-chart", "radarchart"}:
        return _card_wrap(title, _chart_component("radar-chart", section_spec, class_metadata))
    if section_type in {"chart"}:
        return _card_wrap(title, _chart_component("bar-chart", section_spec, class_metadata))
    if section_type in {"dashboard"}:
        return _dashboard_component(section_spec, class_metadata)
    if section_type in {"metric_card", "metric-card", "metric_cards", "kpi", "metric"}:
        return _metric_card_component(section_spec, class_metadata)

    return _content_component(title, body)


# ---------------------------------------------------------------------------
# Phase 3 — LLM-authored HTML sections + structured widget binding
# ---------------------------------------------------------------------------
#
# A section may now be authored three ways (checked in this order):
#   1. ``bind``  — a structured data binding; the server builds the typed,
#                  recognizer-compatible widget and splices it into optional
#                  LLM-authored HTML *chrome* at a ``<!--WIDGET:slot-->`` marker.
#   2. ``html``  — rich themed markup the LLM wrote using the .ds-* classes;
#                  converted to a component tree and normalized to a single
#                  section carrying a stable class + leading heading.
#   3. legacy    — today's typed ``type``/``fields`` builders (full back-compat).
# Every branch degrades gracefully: a failure falls back to the typed builder
# for the declared kind or a themed content box — a section NEVER crashes the
# whole page.

_HEADING_TAGS = ("h1", "h2", "h3")

# Chart ``bind.kind`` -> the chart-type token the typed builder expects.
_BIND_CHART_KINDS = {
    "bar_chart": "bar-chart",
    "pie_chart": "pie-chart",
    "line_chart": "line-chart",
    "radar_chart": "radar-chart",
}


def _section_has_heading(node: Dict[str, Any]) -> bool:
    """True if *node* contains an h1/h2/h3 with text (recursively).

    Mirrors ``_section_label``'s heading probe so an injected heading is
    findable by the modification edit-ops (_match_section / _set_section_heading).
    """
    if not isinstance(node, dict):
        return False
    comps = node.get("components")
    if not isinstance(comps, list):
        return False
    for child in comps:
        if not isinstance(child, dict):
            continue
        if child.get("tagName") in _HEADING_TAGS and _clean_text(child.get("content")):
            return True
        if _section_has_heading(child):
            return True
    return False


def _ensure_section_class(node: Dict[str, Any], fallback: str = "ds-section") -> None:
    """Guarantee *node* carries a non-empty ``class`` so edit-ops can locate it."""
    attrs = node.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
        node["attributes"] = attrs
    if not _clean_text(attrs.get("class")):
        attrs["class"] = fallback


def _is_chrome_section(node: Dict[str, Any]) -> bool:
    """True for full-width chrome (footer / nav) that locates by class, not a
    heading — so we don't inject a spurious filler heading into it."""
    tag = node.get("tagName", "")
    if tag in ("footer", "nav"):
        return True
    attrs = node.get("attributes")
    cls = attrs.get("class", "") if isinstance(attrs, dict) else ""
    classes = set(cls.split()) if isinstance(cls, str) else set()
    return bool(classes & {"ds-footer", "ds-nav", "assistant-footer", "assistant-nav-header"})


def _prepend_heading(node: Dict[str, Any], title: str) -> None:
    """Insert a leading ``<h2 class="ds-heading">`` so the section has a title."""
    heading = {
        "tagName": "h2",
        "attributes": {"class": "ds-heading"},
        "content": _clean_text(title, fallback="Section"),
    }
    # A section whose only child was text is stored via ``content`` — convert it
    # back to a textnode child so we can prepend the heading before it.
    if "content" in node and not isinstance(node.get("components"), list):
        node["components"] = [{"type": "textnode", "content": node.pop("content")}]
    comps = node.get("components")
    if not isinstance(comps, list):
        comps = []
        node["components"] = comps
    comps.insert(0, heading)


def _normalize_html_section(
    nodes: List[Dict[str, Any]],
    title: str = "Section",
    class_hint: str = "ds-section",
) -> Optional[Dict[str, Any]]:
    """Collapse a converted-HTML node list into ONE stable section node.

    Guarantees exactly one top-level node with a class + a leading heading so
    downstream edit-ops (which treat one section == one node) keep matching.
    Returns ``None`` when there is nothing renderable (caller then falls back).
    """
    real = [n for n in nodes if isinstance(n, dict)]
    if not real:
        return None
    if len(real) == 1 and _clean_text(real[0].get("tagName")):
        section = real[0]
    else:
        section = {"tagName": "section", "attributes": {"class": class_hint}, "components": real}
    _ensure_section_class(section, class_hint)
    # Only inject a heading when we have a REAL title. Prepending a generic
    # "Section" (the old fallback) leaked ugly placeholder headings onto every
    # heading-less band (stats rows, CTA bands, image strips). Edit-op matching
    # relies on the stable class, not a forced heading, so skipping is safe.
    clean_title = _clean_text(title)
    if (
        clean_title
        and clean_title.lower() != "section"
        and not _is_chrome_section(section)
        and not _section_has_heading(section)
    ):
        _prepend_heading(section, clean_title)
    return section


def _ds_card_section(title: str, widget_node: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a bare widget in a themed ``.ds-section > .ds-card`` with a heading."""
    card_children: List[Dict[str, Any]] = [
        {
            "tagName": "h3",
            "attributes": {"class": "ds-heading"},
            "content": _clean_text(title, fallback="Section"),
        },
        widget_node,
    ]
    return {
        "tagName": "section",
        "attributes": {"class": "ds-section"},
        "components": [
            {"tagName": "div", "attributes": {"class": "ds-card"}, "components": card_children}
        ],
    }


def _bind_default_title(bind: Dict[str, Any]) -> str:
    return _clean_text(bind.get("kind"), fallback="Data").replace("_", " ").title()


def _build_bind_widget(
    bind: Dict[str, Any],
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build the typed, recognizer-compatible widget node for ``bind.kind``.

    Synthesizes a legacy-style section spec (className / sampleData / fields)
    the existing typed builders understand, so the widget keeps its exact
    recognizer contract (type:'table'+data-source+columns, type:'<chart>'+series,
    type:'metric-card', ...) while the surrounding skin is LLM-authored.
    """
    kind = _clean_text(bind.get("kind")).lower()
    title = _clean_text(section_spec.get("title"), fallback="")
    cta = _clean_text(section_spec.get("ctaLabel"), fallback="Submit")
    class_name = _clean_text(bind.get("className")) or _clean_text(section_spec.get("className"))
    columns = [c for c in (bind.get("columns") or []) if isinstance(c, str) and c.strip()]
    sample = bind.get("sampleData") or section_spec.get("sampleData") or []

    widget_spec: Dict[str, Any] = {
        "title": title,
        "className": class_name,
        "sampleData": sample,
        "fields": columns,
        "rows": bind.get("rows") or [],
    }

    if kind in _BIND_CHART_KINDS:
        return _chart_component(_BIND_CHART_KINDS[kind], widget_spec, class_metadata)
    if kind == "table":
        return _table_component(widget_spec, class_metadata)
    if kind == "metric_card":
        return _metric_card_component(widget_spec, class_metadata)
    if kind == "form":
        return _form_component(title or "Form", columns, cta)
    if kind == "dashboard":
        return _dashboard_component(widget_spec, class_metadata)
    # Unknown kind — a data table is the safest generic widget.
    return _table_component(widget_spec, class_metadata)


def _build_bound_section(
    section_spec: Dict[str, Any],
    bind: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Splice a typed widget into optional LLM chrome, or card-wrap it."""
    widget_node = _build_bind_widget(bind, section_spec, class_metadata)
    title = _clean_text(section_spec.get("title")) or _bind_default_title(bind)

    chrome = section_spec.get("html")
    if _clean_text(chrome):
        nodes = html_to_components(chrome)
        slots = find_widget_slots(nodes)
        if slots:
            nodes = replace_widget_slot(nodes, slots[0], widget_node)
        else:
            # Chrome without a marker: keep the authored skin AND append the
            # widget in a card so its data still renders (nothing lost).
            nodes = list(nodes) + [
                {"tagName": "div", "attributes": {"class": "ds-card"}, "components": [widget_node]}
            ]
        section = _normalize_html_section(nodes, title=title)
        if section is not None:
            return section

    # No chrome (or empty conversion): wrap the widget in a themed card section.
    return _ds_card_section(title, widget_node)


def _build_section_component(
    section_spec: Dict[str, Any],
    class_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Turn one section spec into ONE GrapesJS section node (Phase 3 dispatch).

    Branches: ``bind`` -> typed widget in LLM chrome; ``html`` -> authored
    themed markup; else -> legacy typed builder. Each branch is guarded so any
    failure degrades to the typed builder for the declared kind or a themed
    content box — a section is NEVER allowed to crash the page.
    """
    if not isinstance(section_spec, dict):
        return _content_component("New Section", "Section content")

    bind = section_spec.get("bind")
    html = section_spec.get("html")

    # -- (1) DATA section: structured binding (+ optional LLM chrome) --------
    if isinstance(bind, dict) and _clean_text(bind.get("kind")):
        try:
            return _build_bound_section(section_spec, bind, class_metadata)
        except Exception:
            logger.warning(
                "[GUINoCode] bind section failed; falling back to typed builder",
                exc_info=True,
            )
            fallback_spec = dict(section_spec)
            fallback_spec["type"] = _clean_text(bind.get("kind")) or "content"
            if not _clean_text(fallback_spec.get("className")):
                fallback_spec["className"] = _clean_text(bind.get("className"))
            try:
                return _legacy_section_component(fallback_spec, class_metadata)
            except Exception:
                return _content_component(
                    _clean_text(section_spec.get("title"), fallback="Section"), ""
                )

    # -- (2) HTML section: LLM-authored themed markup -----------------------
    if _clean_text(html):
        try:
            nodes = html_to_components(html)
            section = _normalize_html_section(
                nodes, title=_clean_text(section_spec.get("title"), fallback="Section")
            )
            if section is not None:
                return section
        except Exception:
            logger.warning(
                "[GUINoCode] html section failed; falling back to content box",
                exc_info=True,
            )
        # Empty / unparseable html → themed content box (never crash).
        return _content_component(
            _clean_text(section_spec.get("title"), fallback="Section"),
            _clean_text(section_spec.get("body"), fallback="Section content"),
        )

    # -- (3) Legacy typed section (full back-compat) ------------------------
    try:
        return _legacy_section_component(section_spec, class_metadata)
    except Exception:
        logger.warning(
            "[GUINoCode] legacy section build failed; content fallback",
            exc_info=True,
        )
        return _content_component(
            _clean_text(section_spec.get("title"), fallback="Section"),
            _clean_text(section_spec.get("body"), fallback="Section content"),
        )


def _collect_data_uri_assets(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect any embedded ``data:`` image sources as GrapesJS asset entries.

    LLM markup may embed inline images as data-URIs (external URLs are stripped
    by the converter). Surfacing them in ``model['assets']`` lets the editor's
    asset manager list them. Returns ``[]`` when there is no inline imagery.
    """
    found: List[Dict[str, Any]] = []
    seen: set = set()

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("tagName") == "img":
            attrs = node.get("attributes")
            src = attrs.get("src", "") if isinstance(attrs, dict) else ""
            if isinstance(src, str) and src.startswith("data:") and src not in seen:
                seen.add(src)
                found.append({"type": "image", "src": src})
        _walk(node.get("components"))
        _walk(node.get("frames"))
        comp = node.get("component")
        if isinstance(comp, dict):
            _walk(comp)

    _walk(pages)
    return found


def _extract_balanced_objects(text: str, start: int) -> List[str]:
    """Return each brace-balanced ``{...}`` object found in an array body.

    Scans from ``start`` (the index just after the array's opening ``[``),
    respecting string literals and escapes so braces/brackets inside string
    values never corrupt the depth count. Stops at the array's closing ``]``
    or when the text ends (truncation). A final, unbalanced object — the one
    the LLM was mid-way through when it was cut off — is dropped.
    """
    objects: List[str] = []
    depth = 0
    obj_start = -1
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start != -1:
                    objects.append(text[obj_start:i + 1])
                    obj_start = -1
        elif ch == "]" and depth == 0:
            break
    return objects


def _salvage_truncated_system(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort recovery of a truncated complete-system JSON.

    When the LLM's multi-page JSON is cut off mid-stream (finish_reason ==
    'length'), ``json.loads`` fails and the whole system would otherwise
    collapse to the Welcome stub. This keeps every page that was *fully*
    emitted before the truncation point by scanning the ``pages`` array and
    extracting the balanced page objects, discarding only the final partial
    page. Returns a ``{"projectName", "pages"}`` dict when at least one
    complete page survives, else ``None`` (nothing recoverable → let the
    caller fall back to the stub).
    """
    if not text or '"pages"' not in text:
        return None
    key_idx = text.find('"pages"')
    bracket_idx = text.find("[", key_idx)
    if bracket_idx == -1:
        return None

    parsed_pages: List[Dict[str, Any]] = []
    for obj_str in _extract_balanced_objects(text, bracket_idx + 1):
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed_pages.append(obj)

    if not parsed_pages:
        return None

    project_name = "App"
    m = re.search(r'"projectName"\s*:\s*"([^"]*)"', text)
    if m:
        project_name = m.group(1)

    logger.warning(
        "[GUINoCode] Salvaged %d complete page(s) from truncated system JSON "
        "(kept them instead of collapsing to the Welcome stub).",
        len(parsed_pages),
    )
    return {"projectName": project_name, "pages": parsed_pages}


class GUINoCodeDiagramHandler(BaseDiagramHandler):
    """Handler for GUI no-code diagram generation."""

    def get_diagram_type(self) -> str:
        return "GUINoCodeDiagram"

    def get_system_prompt(self, class_info: str = "") -> str:
        class_block = f"\n\n{class_info}" if class_info else ""
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

    def _parse_page_spec(
        self,
        spec: Dict[str, Any],
        class_metadata: Optional[List[Dict[str, Any]]] = None,
        all_page_names: Optional[List[str]] = None,
        project_name: str = "BESSER",
        tokens: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        page_name = _sanitize_page_name(spec.get("name"), fallback="Page")
        raw_sections = spec.get("sections") if isinstance(spec.get("sections"), list) else []
        sections = [item for item in raw_sections if isinstance(item, dict)]

        wrapper = _default_wrapper_component(tokens)

        # Inject a navigation header bar at the top of every page
        page_components: List[Dict[str, Any]] = []
        if all_page_names and len(all_page_names) > 1:
            page_components.append(
                _nav_header_component(
                    page_names=all_page_names,
                    active_page=page_name,
                    project_name=project_name,
                    tokens=tokens,
                )
            )

        page_components.extend(
            _build_section_component(section, class_metadata)
            for section in sections
        )

        # Separate full-width components (hero, footer, nav) from card sections.
        # Card sections are wrapped in a <main> container with max-width for
        # a clean centered layout. Covers both the legacy ``assistant-*`` skin
        # and the Phase-3 ``ds-*`` design-system classes; the class attribute
        # may carry several classes, so match on any token.
        final_components: List[Dict[str, Any]] = []
        main_children: List[Dict[str, Any]] = []
        _FULL_WIDTH_CLASSES = {
            "assistant-hero", "assistant-footer", "assistant-nav-header",
            "ds-hero", "ds-footer", "ds-nav",
        }

        def _is_full_width(comp: Dict[str, Any]) -> bool:
            attrs = comp.get("attributes")
            cls = attrs.get("class", "") if isinstance(attrs, dict) else ""
            classes = set(cls.split()) if isinstance(cls, str) else set()
            tag = comp.get("tagName", "")
            return bool(classes & _FULL_WIDTH_CLASSES) or tag in ("nav", "footer")

        for comp in page_components:
            if _is_full_width(comp):
                # Flush accumulated main children first
                if main_children:
                    final_components.append(_main_container(main_children))
                    main_children = []
                final_components.append(comp)
            else:
                main_children.append(comp)

        if main_children:
            final_components.append(_main_container(main_children))

        wrapper["components"] = final_components

        return {
            "name": page_name,
            "route_path": f"/{re.sub(r'[^a-z0-9-]+', '-', page_name.lower()).strip('-') or 'page'}",
            "frames": [{"component": wrapper}],
        }

    def _append_section(self, model: Dict[str, Any], page_name: str, section_component: Dict[str, Any]) -> Dict[str, Any]:
        pages = model.get("pages") if isinstance(model.get("pages"), list) else []
        if not pages:
            pages = _default_gui_model()["pages"]
            model["pages"] = pages

        target_page = None
        normalized_target = page_name.lower().strip()
        for page in pages:
            if not isinstance(page, dict):
                continue
            if _clean_text(page.get("name")).lower() == normalized_target:
                target_page = page
                break

        if target_page is None:
            target_page = {
                "name": _sanitize_page_name(page_name, fallback="Page"),
                "route_path": f"/{re.sub(r'[^a-z0-9-]+', '-', page_name.lower()).strip('-') or 'page'}",
                "frames": [{"component": _default_wrapper_component()}],
            }
            pages.append(target_page)

        wrapper = _ensure_page_wrapper(target_page)
        components = wrapper.get("components")
        if not isinstance(components, list):
            components = []
            wrapper["components"] = components
        components.append(section_component)
        return model

    # ------------------------------------------------------------------
    # Message Builders
    # ------------------------------------------------------------------

    def _build_gui_system_message(self, pages: list) -> str:
        """Build a descriptive message for a complete GUI model."""
        page_names = [p.get("name", "Page") for p in pages[:5] if isinstance(p, dict)]
        msg = f"I built **{len(pages)}** screen(s) for your app"
        if page_names:
            msg += f": {', '.join(f'**{n}**' for n in page_names)}"
            if len(pages) > 5:
                msg += f" (+{len(pages) - 5} more)"
        msg += ". Want me to add more screens or change the layout?"
        return msg

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        class_metadata: Optional[List[Dict[str, Any]]] = kwargs.get("class_metadata")
        class_info = ""
        if class_metadata:
            class_info = format_class_metadata_for_prompt(class_metadata)
        prompt = self.get_system_prompt(class_info=class_info)

        try:
            # Single element → SMALL generation tier (latency-sensitive).
            parsed = self.predict_structured(
                f"User Request: {user_request}",
                SingleGUIElementSpec,
                system_prompt=prompt,
                model=MODEL_GENERATION_SMALL,
            )
            spec = parsed.model_dump()

            page_name = _sanitize_page_name(spec.get("pageName"), fallback="Home")
            section_spec = spec.get("section") if isinstance(spec.get("section"), dict) else {}
            section_component = _build_section_component(section_spec, class_metadata)

            model = _default_gui_model()
            model = self._append_section(model, page_name, section_component)
            return {
                "action": "inject_element",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": f"Added a new UI section to the **{page_name}** page.",
            }
        except LLMPredictionError:
            logger.error("[GUINoCode] generate_single_element LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that GUI element. Please try again or rephrase your request.")
        except Exception:
            return self.generate_fallback_element(user_request)

    def _structured_max_tokens(self, response_schema) -> int:
        # Multi-page authored GUIs (and batches carrying authored HTML) need
        # the same extended budget the free-text path used — the shared LARGE
        # tier truncates real multi-page apps.
        if response_schema.__name__ in ("AuthoredSystemGUISpec", "GUIModificationBatchSpec"):
            return GUI_COMPLETE_SYSTEM_MAX_TOKENS
        return super()._structured_max_tokens(response_schema)

    def generate_complete_system(
        self,
        user_request: str,
        existing_model: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        class_metadata: Optional[List[Dict[str, Any]]] = kwargs.get("class_metadata")
        class_block = ""
        if class_metadata:
            class_block = "\n\n" + format_class_metadata_for_prompt(class_metadata)

        # Pick the design domain up front (deterministic keyword heuristic). It
        # drives BOTH the exemplars injected into the authoring prompt below and
        # the final stylesheet at assembly; the LLM may override it via a
        # top-level "domain" field in its output.
        domain = pick_domain(user_request)
        exemplars = block_exemplars(domain)
        exemplars_block = "\n\n".join(
            f"<!-- {name} -->\n{markup}" for name, markup in exemplars.items()
        )

        system_prompt = f"""You are a senior product designer AUTHORING a themed, production-realistic web app for the **{domain}** domain.

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
      <div class='ds-card'>{_SVG_THUMB_HINT}<h3 class='ds-heading'>Ava, 27</h3><p>Loves hiking & jazz. 92% match.</p><a class='ds-btn ds-btn-primary' href='#'>View</a></div>
      ...one card per item, each with a real name/label + concrete copy + (optionally) a visual...
    </div></section>

Imagery (use it where it strengthens the design — heroes, profile/product cards, feature icons):
- ONLY an inline <svg ...>...</svg> or an <img> whose src is a data: URI ever renders — external/http image URLs are blocked by CSP, so never use them.
- Reusable, CSP-safe placeholder if you want one: {_SVG_THUMB_HINT}

Design judgment — build what THIS request actually needs; do not pad or force a fixed skeleton:
1. Pages: let the request set the count. A multi-screen app / platform / dashboard naturally spans several pages (an overview/home plus a screen per major feature); a "landing page" or single-"page" request is one focused page. Each page a coherent set of ordered sections.
2. Lead each page with the section that fits its purpose — a marketing ds-hero for a landing page; a data screen with a heading + KPIs or a ds-notice. Don't default to a hero for a data app.
3. Full-width sections (ds-hero, ds-footer, ds-nav) span the page; other sections sit in cards inside a centered container.
4. Close with a ds-footer when it suits the page (most marketing/landing pages); an app/data screen may not need one.
5. Populate every widget you place, and give each page enough real copy to read as finished — but include only what the request calls for, not filler.
6. VISUAL HIERARCHY: open every page with a page-header moment (title + one-line subtitle + the page's primary action). Vary the rhythm — full-width bands, grids, cards, accent moments — generous whitespace; NEVER a monotone stack of identical white cards. Commit ONE signature visual moment per page (a gradient band, an oversized stat, a colored panel) — safe-but-flat everywhere is a failure.
7. SAMPLE DATA REALISM: figures must be plausible and VARIED — a chart where every value is 1 (or several series with identical data) is a FAILURE. Write numbers a real business would show; one meaningful series unless comparing series is the point. A section whose heading promises a chart / distribution / trend / breakdown MUST be a DATA section with a chart bind carrying that varied sampleData — never a table and never plain html. Headings are never generic labels like 'Table' or 'Section'.
8. Keep an entity's field names IDENTICAL across every section it appears in.
9. Return JSON only.{class_block}"""

        logger.info(f"[GUINoCode] generate_complete_system called with: {user_request[:120]!r}")

        try:
            # --- Two-pass generation for richer UI design ---
            reasoning_prompt = (
                "You are a UI/UX design expert. Think step by step about the "
                "following web application request and plan the page layout.\n\n"
                f"User Request: {user_request}\n\n"
                "Work in this order:\n"
                "1. DATA MODEL FIRST. Identify the 2-4 core entities this app "
                "revolves around. For EACH entity write its exact field names + "
                "types, e.g. Book: title:str, author:str, genre:str, year:int, "
                "available:bool. These field names are the contract for the whole "
                "UI — pick them once and do not vary them.\n"
                "2. Pages: decide how many screens THIS request needs — a full "
                "app/platform/dashboard spans several (an overview/home plus a "
                "screen per major feature or entity); a 'landing page' or single "
                "'page' request is one focused page. Don't pad.\n"
                "3. Sections per page: choose the app TYPE (marketing / dashboard / "
                "data-catalog / CRUD) and lead each page with the section that fits "
                "it — do NOT default to a marketing hero for a data app.\n"
                "4. Binding (most important): every table/chart/form that shows an "
                "entity MUST reuse the EXACT field names from step 1 — the same "
                "entity must look identical on every page. A chart aggregates ONE "
                "field (e.g. count of books by genre).\n"
                "5. Navigation flow, and realistic domain sample data built from the "
                "step-1 fields.\n"
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

            # Complete-system generation → LARGE tier; reasoning pass on
            # the REASONING tier (see model_config).
            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                response_schema=AuthoredSystemGUISpec,
                model=MODEL_GENERATION_LARGE,
                reasoning_model=MODEL_REASONING,
            )
            spec = parsed.model_dump()
            # Schema enforcement replaces the old clean/parse/salvage chain.
            # Drop any section carrying neither authored html nor a binding.
            for page in spec.get("pages", []):
                page["sections"] = [
                    sc for sc in page.get("sections", [])
                    if isinstance(sc, dict)
                    and (
                        _clean_text(sc.get("html"))
                        or (
                            isinstance(sc.get("bind"), dict)
                            and _clean_text((sc.get("bind") or {}).get("kind"))
                        )
                    )
                ]

            pages_spec = spec.get("pages") if isinstance(spec.get("pages"), list) else []
            project_name = spec.get("projectName", "App")
            all_page_names = [
                _clean_text(p.get("name")) or f"Page{i}"
                for i, p in enumerate(pages_spec, 1)
                if isinstance(p, dict)
            ]

            # Resolve the theme FIRST — the nav header and page wrapper are
            # assembled from these tokens, so the whole page (not just the
            # ds-* sections) wears the app's actual identity.
            spec_domain = _clean_text(spec.get("domain")) or domain
            theme_overrides = spec.get("theme") if isinstance(spec.get("theme"), dict) else None
            try:
                if theme_overrides and any(theme_overrides.values()):
                    tokens = merge_theme_overrides(spec_domain, theme_overrides)
                else:
                    tokens = theme_tokens(spec_domain)
                styles = stylesheet_rules_from_tokens(tokens)
            except Exception:
                logger.warning(
                    "[GUINoCode] stylesheet_rules failed; falling back to empty styles",
                    exc_info=True,
                )
                tokens = None
                styles = []

            # The LLM's own stylesheet (custom app-* classes) layers on top of
            # the design-system baseline — its creative freedom beyond the kit.
            try:
                llm_rules = _parse_llm_css(spec.get("css"))
                if llm_rules:
                    styles = styles + llm_rules
                    logger.info(
                        "[GUINoCode] LLM stylesheet accepted (%d rules)", len(llm_rules)
                    )
            except Exception:
                logger.warning("[GUINoCode] LLM stylesheet parse failed", exc_info=True)

            pages = [
                self._parse_page_spec(
                    page,
                    class_metadata,
                    all_page_names=all_page_names,
                    project_name=project_name,
                    tokens=tokens,
                )
                for page in pages_spec
                if isinstance(page, dict)
            ]
            if not pages:
                return self._error_response(
                    "I couldn't produce any pages for that request — your "
                    "current screens are untouched. Try describing the app "
                    "again with a bit more detail."
                )

            model = {
                "pages": pages,
                "styles": styles,
                "assets": _collect_data_uri_assets(pages),
                "symbols": [],
                "version": DEFAULT_GUI_VERSION,
            }

            return {
                "action": "inject_complete_system",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": self._build_gui_system_message(pages),
            }
        except LLMPredictionError:
            logger.error("[GUINoCode] generate_complete_system LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that GUI. Please try again or rephrase your request.")
        except Exception:
            logger.error(
                "[GUINoCode] generate_complete_system assembly FAILED", exc_info=True
            )
            return self._error_response(
                "Something went wrong while assembling that GUI — your current "
                "screens are untouched. Please try again."
            )

    def generate_modification(
        self,
        user_request: str,
        current_model: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        class_metadata: Optional[List[Dict[str, Any]]] = kwargs.get("class_metadata")
        class_block = ""
        if class_metadata:
            class_block = "\n\n" + format_class_metadata_for_prompt(class_metadata)

        model = _normalize_gui_model(current_model)
        page_names = [
            _clean_text(page.get("name"))
            for page in model.get("pages", [])
            if isinstance(page, dict) and _clean_text(page.get("name"))
        ]
        pages_hint = ", ".join(page_names) if page_names else "Home"

        # Prefer the user's raw message for deterministic parsing — the modeling
        # prompt passed in ``user_request`` is enriched with workspace context.
        raw_request = _clean_text(kwargs.get("raw_request")) or user_request

        # ── 1. Deterministic fast-path ──────────────────────────────────────
        # Simple, unambiguous edits (rename / recolor / reorder) are applied
        # directly without an LLM round-trip. This is fast AND robust: it can
        # never fail the way a structured-output call can, and never destroys
        # the existing model.
        try:
            fast = self._try_deterministic_modify(model, raw_request, page_names)
            if fast is not None:
                applied_model, message = fast
                return {
                    "action": "modify_model",
                    "diagramType": self.get_diagram_type(),
                    "model": applied_model,
                    "message": message,
                }
        except Exception:
            logger.warning("[GUINoCode] deterministic modify fast-path failed", exc_info=True)

        # ── 2. LLM-structured path ──────────────────────────────────────────
        outline = self._page_outline(model)
        prompt = f"""You are the design agent for an existing web GUI. Apply the user's requested edit(s) precisely and surgically — change ONLY what they asked for.

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

Rules:
1. Choose the most specific operation — never append_section for a rename/recolor/reorder/remove/edit request.
2. Use page names and section headings EXACTLY as listed above.
3. edit_section replaces the WHOLE section: carry over what should stay, change what was asked.
4. Keep an entity's field names identical to how they appear elsewhere in the app.{class_block}"""

        try:
            user_prompt = f"User Request: {raw_request}"
            # Modification → SMALL generation tier (latency-sensitive); the
            # batch schema rides the LARGE token budget for authored HTML.
            parsed = self.predict_structured(
                user_prompt, GUIModificationBatchSpec, system_prompt=prompt,
                model=MODEL_GENERATION_SMALL,
            )
            operations = parsed.model_dump().get("operations") or []
            messages: List[str] = []
            for op in operations[:5]:
                if not isinstance(op, dict):
                    continue
                # Recompute page names between ops — an earlier add/rename
                # must be visible to the next op in the same batch.
                page_names = [
                    _clean_text(page.get("name"))
                    for page in model.get("pages", [])
                    if isinstance(page, dict) and _clean_text(page.get("name"))
                ]
                model, message = self._apply_modification_spec(
                    model, op, page_names, class_metadata,
                )
                messages.append(message)
            if not messages:
                raise ValueError("empty modification batch")
            combined = (
                messages[0] if len(messages) == 1
                else "\n".join(f"- {m}" for m in messages)
            )
            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": combined,
            }
        except LLMPredictionError:
            logger.error("[GUINoCode] generate_modification LLM FAILED", exc_info=True)
            # SAFETY: never destroy the existing GUI on failure. Return the
            # original model unchanged with a clarification asking the user to
            # be specific, rather than an empty/error payload.
            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": (
                    "I couldn't apply that change automatically, and I've left your GUI "
                    "unchanged. Could you be more specific? For example: *'rename the "
                    f"{(page_names[0] if page_names else 'Home')} page to Overview'*, "
                    "*'change the hero section color to red'*, or *'move the Recent edits "
                    "card to the top'*."
                ),
            }
        except Exception:
            logger.warning("[GUINoCode] generate_modification apply failed", exc_info=True)
            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": (
                    "I couldn't parse that GUI modification, but your existing model is "
                    "safe and unchanged. Could you rephrase what you'd like to change?"
                ),
            }

    # ------------------------------------------------------------------
    # Modification helpers
    # ------------------------------------------------------------------

    def _try_deterministic_modify(
        self,
        model: Dict[str, Any],
        request: str,
        page_names: List[str],
    ) -> Optional[tuple]:
        """Apply common edits parsed directly from the request, no LLM needed.

        Returns ``(model, message)`` when a rule fired, else ``None`` so the
        caller falls through to the LLM path. Never deletes content.
        """
        text = _clean_text(request)
        if not text:
            return None
        # Trailing punctuation ("...to red?") would otherwise fall outside the
        # color/name char classes and break the end-anchored patterns.
        text = text.rstrip("?.! ").strip()
        low = text.lower()

        # --- Rename a page: "rename X to Y" / "rename page X to Y" ---
        # Capture everything between "rename [page]" and "to" as the source so
        # multi-word names ("Page Management") survive, then resolve the source
        # against existing page names (most robust) before falling back to a
        # section rename.
        m = re.search(
            r"rename\s+(?:the\s+)?(?:page\s+)?['\"]?(.+?)['\"]?\s+to\s+['\"]?(.+?)['\"]?$",
            text, re.IGNORECASE,
        )
        if m:
            raw_old = _clean_text(m.group(1))
            new_name = _sanitize_page_name(m.group(2))
            # Prefer the existing page name that the source text matches, so a
            # leading word ("page") swallowed by the regex doesn't break it.
            matched_page = None
            _old = raw_old.lower()
            # Exact match across ALL pages first, so "Management" can't hijack
            # "User Management" just because it's earlier in the list (#58).
            for page in model.get("pages", []):
                if isinstance(page, dict) and _clean_text(page.get("name")).lower() == _old:
                    matched_page = page
                    break
            if matched_page is None:
                # Fall back to a suffix match only when it is UNAMBIGUOUS.
                suffix_matches = [
                    page for page in model.get("pages", [])
                    if isinstance(page, dict)
                    and _clean_text(page.get("name"))
                    and _clean_text(page.get("name")).lower().endswith(_old)
                ]
                if len(suffix_matches) == 1:
                    matched_page = suffix_matches[0]
            if matched_page is not None:
                old_name = _clean_text(matched_page.get("name"))
                matched_page["name"] = new_name
                matched_page["route_path"] = (
                    f"/{re.sub(r'[^a-z0-9-]+', '-', new_name.lower()).strip('-') or 'page'}"
                )
                return model, f"Renamed the **{old_name}** page to **{new_name}**."
            # Otherwise it may be a section rename — try that.
            if self._rename_section(model, raw_old, new_name):
                return model, f"Renamed the **{raw_old}** section to **{new_name}**."

        # --- Recolor: "change [the] [<target>] color to <color>" ---
        # Require the word "colo(u)r" to be present so we don't misread an
        # open-ended request ("make the dashboard fancier") as a recolor.
        m = None
        color_word = None
        if re.search(r"\bcolou?r\b", low):
            m = re.search(
                r"(?:change|set|make|turn|paint)\b.*?\bcolou?r\b.*?\bto\b\s+([a-z#0-9(),.\s]+)$",
                low,
            )
            if not m:
                # "make the hero red" / "make it blue" — only when the trailing
                # word is an actual named color.
                m = re.search(r"\bmake\b\s+(?:it|the\s+.+?)\s+([a-z]+)$", low)
                if m and m.group(1) not in _NAMED_COLORS:
                    m = None
        else:
            # "make the hero red" without the word "color": accept only if the
            # final token is a recognised color name.
            trailing = re.search(r"\b(?:make|turn|paint)\b\s+(?:it|the\s+.+?)\s+([a-z]+)$", low)
            if trailing and trailing.group(1) in _NAMED_COLORS:
                m = trailing
        if m:
            color_word = m.group(1).strip()
            color = _resolve_color(color_word)
            if color and (color_word in _NAMED_COLORS or color.startswith(("#", "rgb", "hsl"))):
                # Did the user name a specific section?
                section_ref = self._extract_section_reference(low)
                if section_ref:
                    changed = self._recolor_section(model, page_names, section_ref, color)
                    if changed:
                        return model, f"Changed the **{section_ref}** section color to **{color_word}**."
                # Otherwise recolor the page(s).
                self._recolor_page(model, page_names, color)
                return model, f"Changed the GUI color to **{color_word}**."

        # --- Reorder / align: "move <section> to the top/bottom" ---
        m = re.search(
            r"(?:move|put|place|align)\s+(?:the\s+)?['\"]?(.+?)['\"]?\s+(?:card|section)?\s*(?:to|at|on)\s+(?:the\s+)?(top|bottom|first|last|start|end)\b",
            low,
        )
        if m:
            section_ref = _clean_text(m.group(1))
            pos_raw = m.group(2)
            position = "top" if pos_raw in ("top", "first", "start") else "bottom"
            moved = self._reorder_section(model, page_names, section_ref, position)
            if moved:
                return model, f"Moved the **{section_ref}** section to the **{position}** of the page."

        return None

    @staticmethod
    def _extract_section_reference(low_text: str) -> Optional[str]:
        """Pull a section name/type out of a recolor request, if present."""
        m = re.search(
            r"\b(hero|footer|header|table|form|content|dashboard|feature[s]?|nav(?:igation)?|sidebar|"
            r"card|stats?|chart|banner)\b",
            low_text,
        )
        return m.group(1) if m else None

    def _rename_section(self, model: Dict[str, Any], query: str, new_title: str) -> bool:
        for page in model.get("pages", []):
            if not isinstance(page, dict):
                continue
            wrapper = _ensure_page_wrapper(page)
            for _parent, _idx, comp in _iter_section_components(wrapper):
                if _match_section(comp, query):
                    if _set_section_heading(comp, new_title):
                        return True
        return False

    def _recolor_section(
        self, model: Dict[str, Any], page_names: List[str], query: str, color: str,
    ) -> bool:
        changed = False
        for page in model.get("pages", []):
            if not isinstance(page, dict):
                continue
            wrapper = _ensure_page_wrapper(page)
            for _parent, _idx, comp in _iter_section_components(wrapper):
                if _match_section(comp, query):
                    style = comp.get("style")
                    if not isinstance(style, dict):
                        style = {}
                        comp["style"] = style
                    # Heroes use a gradient background; override with a solid color.
                    style.pop("background", None)
                    style["background-color"] = color
                    changed = True
        return changed

    def _recolor_page(self, model: Dict[str, Any], page_names: List[str], color: str) -> None:
        """Set the page background color on every page wrapper."""
        for page in model.get("pages", []):
            if not isinstance(page, dict):
                continue
            wrapper = _ensure_page_wrapper(page)
            style = wrapper.get("style")
            if not isinstance(style, dict):
                style = {}
                wrapper["style"] = style
            style["background-color"] = color

    def _reorder_section(
        self, model: Dict[str, Any], page_names: List[str], query: str, position: str,
    ) -> bool:
        for page in model.get("pages", []):
            if not isinstance(page, dict):
                continue
            wrapper = _ensure_page_wrapper(page)
            for parent, idx, comp in _iter_section_components(wrapper):
                if _match_section(comp, query):
                    # Remove from current spot and re-insert at top/bottom of
                    # the same parent list.
                    parent.pop(idx)
                    if position == "top":
                        parent.insert(0, comp)
                    else:
                        parent.append(comp)
                    return True
        return False

    def _page_outline(self, model: Dict[str, Any]) -> str:
        """Human-readable page/section outline for the modify prompt.

        Gives the LLM exact targets ('Home: Hero | Recent records | Footer')
        so edits land on the right section instead of a guessed one.
        """
        lines: List[str] = []
        for page in model.get("pages", []):
            if not isinstance(page, dict):
                continue
            name = _clean_text(page.get("name")) or "Page"
            labels: List[str] = []
            try:
                wrapper = _ensure_page_wrapper(page)
                for _parent, _idx, comp in _iter_section_components(wrapper):
                    labels.append(_section_label(comp))
            except Exception:
                pass
            lines.append(f"- {name}: " + (" | ".join(labels) if labels else "(empty)"))
        return "\n".join(lines) if lines else "- Home: (empty)"

    def _apply_modification_spec(
        self,
        model: Dict[str, Any],
        spec: Dict[str, Any],
        page_names: List[str],
        class_metadata: Optional[List[Dict[str, Any]]],
    ) -> tuple:
        """Apply a parsed ``GUIModificationSpec`` dict to *model* in place.

        Returns ``(model, message)``. Never deletes the last remaining page.
        """
        operation = _clean_text(spec.get("operation"), fallback="append_section")
        default_page = page_names[0] if page_names else "Home"
        page_name = _sanitize_page_name(spec.get("pageName"), fallback=default_page)

        if operation == "rename_page":
            new_page_name = _sanitize_page_name(spec.get("newPageName"), fallback=page_name)
            renamed = False
            for page in model.get("pages", []):
                if isinstance(page, dict) and _clean_text(page.get("name")).lower() == page_name.lower():
                    page["name"] = new_page_name
                    page["route_path"] = (
                        f"/{re.sub(r'[^a-z0-9-]+', '-', new_page_name.lower()).strip('-') or 'page'}"
                    )
                    renamed = True
                    break
            if renamed:
                return model, f"Renamed the **{page_name}** page to **{new_page_name}**."
            # Don't claim success when no page matched (a real source of
            # "it said it renamed but nothing changed" reports).
            return model, (
                f"I couldn't find a page named **{page_name}** to rename. "
                "Your GUI is unchanged."
            )

        if operation == "add_page":
            new_page_name = _sanitize_page_name(spec.get("newPageName") or page_name, fallback="Page")
            new_page = {
                "name": new_page_name,
                "route_path": f"/{re.sub(r'[^a-z0-9-]+', '-', new_page_name.lower()).strip('-') or 'page'}",
                "frames": [{"component": _default_wrapper_component()}],
            }
            model.setdefault("pages", []).append(new_page)
            section_spec = spec.get("section") if isinstance(spec.get("section"), dict) else None
            if section_spec:
                model = self._append_section(model, new_page_name, _build_section_component(section_spec, class_metadata))
            return model, f"Added a new page named **{new_page_name}**."

        if operation == "remove_page":
            pages = [p for p in model.get("pages", []) if isinstance(p, dict)]
            filtered = [p for p in pages if _clean_text(p.get("name")).lower() != page_name.lower()]
            # SAFETY: never remove the last page — leave at least one.
            if filtered and len(filtered) < len(pages):
                model["pages"] = filtered
                return model, f"Removed the **{page_name}** page from the GUI."
            return model, (
                f"I couldn't remove the **{page_name}** page (it either doesn't exist or "
                "it's the only page). Your GUI is unchanged."
            )

        if operation == "rename_section":
            query = _clean_text(spec.get("sectionTitle"))
            new_title = _clean_text(spec.get("newSectionTitle"))
            if query and new_title and self._rename_section(model, query, new_title):
                return model, f"Renamed the **{query}** section to **{new_title}**."
            return model, (
                f"I couldn't find a section matching **{query or 'that'}** to rename. "
                "Your GUI is unchanged."
            )

        if operation == "recolor_section":
            query = _clean_text(spec.get("sectionTitle"))
            color = _resolve_color(spec.get("color"))
            if query and color and self._recolor_section(model, page_names, query, color):
                return model, f"Changed the **{query}** section color."
            return model, (
                f"I couldn't find a section matching **{query or 'that'}** to recolor. "
                "Your GUI is unchanged."
            )

        if operation == "recolor_page":
            color = _resolve_color(spec.get("color"))
            if color:
                self._recolor_page(model, page_names, color)
                return model, "Changed the GUI background color."
            return model, "I couldn't determine the requested color. Your GUI is unchanged."

        if operation == "reorder_section":
            query = _clean_text(spec.get("sectionTitle"))
            position = _clean_text(spec.get("position"), fallback="top")
            position = "top" if position in ("top", "up", "first", "start") else "bottom"
            if query and self._reorder_section(model, page_names, query, position):
                return model, f"Moved the **{query}** section to the **{position}** of the page."
            return model, (
                f"I couldn't find a section matching **{query or 'that'}** to move. "
                "Your GUI is unchanged."
            )

        if operation == "edit_section":
            query = _clean_text(spec.get("sectionTitle"))
            section_spec = spec.get("section") if isinstance(spec.get("section"), dict) else None
            if query and section_spec:
                replacement = _build_section_component(section_spec, class_metadata)
                for page in model.get("pages", []):
                    if not isinstance(page, dict):
                        continue
                    wrapper = _ensure_page_wrapper(page)
                    for parent, idx, comp in _iter_section_components(wrapper):
                        if _match_section(comp, query):
                            parent[idx] = replacement
                            return model, f"Updated the **{query}** section."
            return model, (
                f"I couldn't find a section matching **{query or 'that'}** to edit. "
                "Your GUI is unchanged."
            )

        if operation == "remove_section":
            query = _clean_text(spec.get("sectionTitle"))
            removed = self._remove_section(model, query)
            if removed:
                return model, f"Removed the **{query}** section."
            return model, (
                f"I couldn't find a section matching **{query or 'that'}** to remove. "
                "Your GUI is unchanged."
            )

        # Default: append_section
        section_spec = spec.get("section") if isinstance(spec.get("section"), dict) else {}
        section_component = _build_section_component(section_spec, class_metadata)
        model = self._append_section(model, page_name, section_component)
        return model, f"Added a new section to the **{page_name}** page."

    def _remove_section(self, model: Dict[str, Any], query: str) -> bool:
        if not query:
            return False
        for page in model.get("pages", []):
            if not isinstance(page, dict):
                continue
            wrapper = _ensure_page_wrapper(page)
            for parent, idx, comp in _iter_section_components(wrapper):
                if _match_section(comp, query):
                    parent.pop(idx)
                    return True
        return False

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        model = _default_gui_model()
        model = self._append_section(
            model,
            "Home",
            _content_component("New Section", "Describe your content and I will refine it."),
        )
        return {
            "action": "inject_element",
            "diagramType": self.get_diagram_type(),
            "model": model,
            "message": "I created a basic GUI section as a starting point. Describe what you'd like (e.g. *'Create a dashboard for shoes with a bar chart and a table'*) and I'll build it!",
        }

    def generate_fallback_system(self) -> Dict[str, Any]:
        model = _default_gui_model()
        model = self._append_section(
            model,
            "Home",
            _hero_component("Welcome", "Start building your interface here.", "Get Started"),
        )
        return {
            "action": "inject_complete_system",
            "diagramType": self.get_diagram_type(),
            "model": model,
            "message": "I created a basic GUI with a welcome page. Describe your app's pages and features for a richer result!",
        }
