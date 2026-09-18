"""Per-domain design system for GUI No-Code generation (Phase 2).

The GUINoCode generator historically ships every app with an empty GrapesJS
``styles[]`` array, so the *only* visual identity a generated app has is the one
hardcoded slate-blue skin baked into a handful of inline component styles. Every
generated GUI therefore looks identical regardless of whether it is a government
service portal or a playful SaaS landing page.

This module gives generation a real, per-domain visual identity. It defines a
small set of **design themes** (``government``, ``finance``, ``health``,
``startup``, ``default``), each a bundle of concrete design *tokens* — palette,
type scale, spacing scale, radius, shadow and a **system-only** font stack — and
derives from those tokens:

1. ``stylesheet_rules(domain)`` — a list of GrapesJS CSS **rule objects** in the
   exact shape the frontend's ``editor.loadProjectData`` consumes in the
   project-data ``styles[]`` array (verified against the WME
   ``templates/pattern/gui/Complete.json`` fixture and
   ``shared/types/project.ts``). Dropping this list into ``model["styles"]`` is
   what actually makes a generated app *look* like its domain.
2. ``block_exemplars(domain)`` — domain-appropriate HTML block snippets built
   from the reusable ``.ds-*`` component classes and semantic text tags. Phase 3
   feeds these to the LLM as proven, editable-safe composition patterns; Phase 4
   splices real data widgets into the ``<!--WIDGET:slot-->`` placeholders.

The module is deliberately **pure and standalone** — it imports nothing from the
handler and has no side effects — so it can be unit-tested in isolation and
reused by the generation prompt builder without dragging in LLM machinery.

CSP note: the editor serves generated apps under a strict Content-Security-Policy
that blocks external stylesheets, webfonts and remote images. Every font stack
here is therefore composed of **system fonts only**, and every piece of imagery
in the exemplars is a CSS gradient or inline SVG — never an ``<img>`` or an
``@font-face``/``<link>`` webfont.
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# System-only font stacks (no external webfonts — CSP blocks @font-face/<link>)
# ---------------------------------------------------------------------------

_SANS = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
_UI = (
    "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Roboto, 'Helvetica Neue', Arial, sans-serif"
)
_SERIF = "'Fraunces', Georgia, Cambria, 'Times New Roman', Times, serif"
_MONO = (
    "ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', "
    "Menlo, monospace"
)

DOMAINS = ("government", "finance", "health", "startup", "default")
DEFAULT_DOMAIN = "default"


# ---------------------------------------------------------------------------
# Design tokens — concrete values per domain
# ---------------------------------------------------------------------------
#
# Each theme is a bundle of concrete token values. The ``.ds-*`` component
# classes and the block exemplars are *derived* from these tokens, so changing a
# token here ripples through the whole generated skin for that domain.

THEMES: Dict[str, Dict] = {
    # Government — restrained navy/slate, high contrast, minimal radius, formal.
    # Serif headings evoke official / civic typography; imagery kept flat.
    "government": {
        "name": "government",
        "palette": {
            "primary": "#1a2b4a",
            "secondary": "#33475b",
            "accent": "#b8860b",
            "surface": "#ffffff",
            "background": "#f4f6f8",
            "text": "#14213d",
            "muted": "#5a6b7b",
            "border": "#c8d1da",
        },
        "type_scale": {
            "h1": {"size": "2.5rem", "weight": "700"},
            "h2": {"size": "1.75rem", "weight": "700"},
            "h3": {"size": "1.125rem", "weight": "600"},
            "body": {"size": "1rem", "weight": "400"},
            "small": {"size": "0.8125rem", "weight": "400"},
        },
        "spacing": {
            "xs": "0.25rem", "sm": "0.5rem", "md": "0.875rem",
            "lg": "1.25rem", "xl": "2rem", "section": "3rem",
        },
        "radius": "2px",
        "shadow": "0 1px 2px rgba(20, 33, 61, 0.10)",
        "line_height": "1.55",
        "font": _SANS,
        "font_heading": _SERIF,
        "font_mono": _MONO,
        "hero_bg": "#1a2b4a",
        "hero_text": "#ffffff",
        "container": "1120px",
    },
    # Finance — deep green/graphite, dense, data-forward. Monospaced KPI figures
    # for a ledger/terminal feel; tighter spacing to pack more data per screen.
    "finance": {
        "name": "finance",
        "palette": {
            "primary": "#0e5c43",
            "secondary": "#1f2933",
            "accent": "#14b8a6",
            "surface": "#ffffff",
            "background": "#eef1f0",
            "text": "#1f2933",
            "muted": "#5f6c66",
            "border": "#d3d9d6",
        },
        "type_scale": {
            "h1": {"size": "2.25rem", "weight": "700"},
            "h2": {"size": "1.625rem", "weight": "650"},
            "h3": {"size": "1.0625rem", "weight": "600"},
            "body": {"size": "0.9375rem", "weight": "400"},
            "small": {"size": "0.8125rem", "weight": "500"},
        },
        "spacing": {
            "xs": "0.25rem", "sm": "0.375rem", "md": "0.75rem",
            "lg": "1rem", "xl": "1.5rem", "section": "2.5rem",
        },
        "radius": "4px",
        "shadow": "0 1px 3px rgba(15, 42, 30, 0.12)",
        "line_height": "1.5",
        "font": _SANS,
        "font_heading": _SANS,
        "font_mono": _MONO,
        "hero_bg": "linear-gradient(135deg, #0e5c43 0%, #1f2933 100%)",
        "hero_text": "#ffffff",
        "container": "1180px",
    },
    # Health — calm teal/white, generous spacing, soft radius. Light, airy hero
    # (dark text on a soft teal wash) and a warm coral accent for friendly CTAs.
    "health": {
        "name": "health",
        "palette": {
            "primary": "#0d8b8b",
            "secondary": "#56c0bd",
            "accent": "#ef8f6e",
            "surface": "#ffffff",
            "background": "#f2f9f9",
            "text": "#1d3a3a",
            "muted": "#5b7a7a",
            "border": "#d3e6e5",
        },
        "type_scale": {
            "h1": {"size": "2.85rem", "weight": "600"},
            "h2": {"size": "2rem", "weight": "600"},
            "h3": {"size": "1.25rem", "weight": "500"},
            "body": {"size": "1.0625rem", "weight": "400"},
            "small": {"size": "0.875rem", "weight": "400"},
        },
        "spacing": {
            "xs": "0.375rem", "sm": "0.75rem", "md": "1.25rem",
            "lg": "2rem", "xl": "3rem", "section": "5rem",
        },
        "radius": "14px",
        "shadow": "0 4px 16px rgba(13, 139, 139, 0.12)",
        "line_height": "1.65",
        "font": _UI,
        "font_heading": _UI,
        "font_mono": _MONO,
        "hero_bg": "linear-gradient(160deg, #e8f7f6 0%, #d2efee 100%)",
        "hero_text": "#1d3a3a",
        "container": "1160px",
    },
    # Startup — vivid gradient accent, large hero type, playful. Big radii, a
    # colourful shadow and a violet→pink gradient carried into hero + primary CTA.
    "startup": {
        "name": "startup",
        "palette": {
            "primary": "#6d28d9",
            "secondary": "#0ea5e9",
            "accent": "#ec4899",
            "surface": "#ffffff",
            "background": "#faf7ff",
            "text": "#1a1130",
            "muted": "#6b6580",
            "border": "#e7e0f5",
        },
        "type_scale": {
            "h1": {"size": "3.5rem", "weight": "800"},
            "h2": {"size": "2.25rem", "weight": "700"},
            "h3": {"size": "1.375rem", "weight": "600"},
            "body": {"size": "1.0625rem", "weight": "400"},
            "small": {"size": "0.875rem", "weight": "500"},
        },
        "spacing": {
            "xs": "0.5rem", "sm": "1rem", "md": "1.5rem",
            "lg": "2.5rem", "xl": "4rem", "section": "6rem",
        },
        "radius": "18px",
        "shadow": "0 10px 30px rgba(109, 40, 217, 0.18)",
        "line_height": "1.6",
        "font": _UI,
        "font_heading": _UI,
        "font_mono": _MONO,
        "gradient": "linear-gradient(135deg, #6d28d9 0%, #ec4899 100%)",
        "hero_bg": "linear-gradient(135deg, #6d28d9 0%, #ec4899 100%)",
        "hero_text": "#ffffff",
        "container": "1200px",
    },
    # Default — clean neutral blue. The safe, product-neutral fallback skin.
    "default": {
        "name": "default",
        "palette": {
            "primary": "#2563eb",
            "secondary": "#475569",
            "accent": "#0ea5e9",
            "surface": "#ffffff",
            "background": "#f8fafc",
            "text": "#0f172a",
            "muted": "#64748b",
            "border": "#e2e8f0",
        },
        "type_scale": {
            "h1": {"size": "2.75rem", "weight": "750"},
            "h2": {"size": "1.875rem", "weight": "650"},
            "h3": {"size": "1.25rem", "weight": "600"},
            "body": {"size": "1rem", "weight": "400"},
            "small": {"size": "0.875rem", "weight": "400"},
        },
        "spacing": {
            "xs": "0.25rem", "sm": "0.5rem", "md": "1rem",
            "lg": "1.5rem", "xl": "2.5rem", "section": "4rem",
        },
        "radius": "8px",
        "shadow": "0 2px 8px rgba(15, 23, 42, 0.08)",
        "line_height": "1.6",
        "font": _SANS,
        "font_heading": _SANS,
        "font_mono": _MONO,
        "hero_bg": "linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%)",
        "hero_text": "#ffffff",
        "container": "1200px",
    },
}


# ---------------------------------------------------------------------------
# Domain selection heuristic
# ---------------------------------------------------------------------------

# Keyword → domain map. Matched case-insensitively with word boundaries so a
# keyword only fires on a whole word ("care" won't match "scared"); multi-word
# phrases ("public sector") are matched as-is. The domain with the most matching
# keywords wins; ties and no-matches fall back to ``default``.
_DOMAIN_KEYWORDS: Dict[str, tuple] = {
    "government": (
        "government", "governmental", "govt", "ministry", "municipal",
        "municipality", "public sector", "citizen", "residency", "resident",
        "certificate", "permit", "license", "licence", "passport", "visa",
        "council", "agency", "civic", "immigration", "welfare", "benefits",
        "court", "official", "tax office", "public service", "e-government",
    ),
    "finance": (
        "finance", "financial", "bank", "banking", "trading", "trade", "trader",
        "invest", "investment", "portfolio", "stock", "stocks", "crypto",
        "fintech", "ledger", "accounting", "invoice", "invoicing", "payment",
        "payments", "revenue", "budget", "expense", "expenses", "wallet",
        "loan", "mortgage", "insurance", "brokerage", "treasury", "billing",
    ),
    "health": (
        "health", "healthcare", "patient", "clinic", "clinical", "hospital",
        "medical", "medicine", "doctor", "appointment", "prescription",
        "pharmacy", "telehealth", "telemedicine", "wellness", "therapy",
        "dental", "nurse", "nursing", "diagnosis", "symptom", "ehr", "vitals",
        "care portal",
    ),
    "startup": (
        "startup", "saas", "landing", "landing page", "launch", "waitlist",
        "mvp", "pitch", "growth", "subscription", "onboarding", "early access",
        "beta", "founder", "founders", "marketing site", "product launch",
        "sign up", "signup", "freemium",
    ),
}


def pick_domain(text: str) -> str:
    """Map a free-text request to one of the design domains.

    A lightweight keyword heuristic: the domain whose vocabulary appears most
    often in ``text`` wins. Returns ``'default'`` when nothing matches (or on
    empty/invalid input).
    """
    if not isinstance(text, str) or not text.strip():
        return DEFAULT_DOMAIN

    lowered = text.lower()
    scores: Dict[str, int] = {domain: 0 for domain in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lowered):
                scores[domain] += 1

    best_domain, best_score = DEFAULT_DOMAIN, 0
    for domain in _DOMAIN_KEYWORDS:  # deterministic priority via declared order
        if scores[domain] > best_score:
            best_domain, best_score = domain, scores[domain]

    return best_domain if best_score > 0 else DEFAULT_DOMAIN


def theme_tokens(domain: str) -> Dict:
    """Return a deep copy of the token bundle for ``domain`` (default fallback)."""
    theme = THEMES.get(domain) or THEMES[DEFAULT_DOMAIN]
    return copy.deepcopy(theme)


# ---------------------------------------------------------------------------
# Rule-object builders (GrapesJS project-data ``styles[]`` shape)
# ---------------------------------------------------------------------------
#
# Verified against templates/pattern/gui/Complete.json:
#   class rule : {"selectors": [{"name": "ds-card"}], "style": {...}}
#   base rule  : {"selectors": [], "selectorsAdd": "*"|"body"|":root", "style": {...}}
#   media rule : {"selectors": [{"name": "ds-grid-2"}], "atRuleType": "media",
#                 "mediaText": "(max-width: 768px)", "style": {...}}


def _class_rule(name: str, style: Dict[str, str]) -> Dict:
    return {"selectors": [{"name": name}], "style": dict(style)}


def _raw_rule(selector: str, style: Dict[str, str]) -> Dict:
    """A base/reset/token/compound rule keyed by a raw selector string."""
    return {"selectors": [], "selectorsAdd": selector, "style": dict(style)}


def _media_class_rule(name: str, media_text: str, style: Dict[str, str]) -> Dict:
    return {
        "selectors": [{"name": name}],
        "atRuleType": "media",
        "mediaText": media_text,
        "style": dict(style),
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba(r, g, b, a)`` string (for token tints).

    Falls back to the source colour unchanged if it is not a 6-digit hex.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return hex_color
    return f"rgba({r}, {g}, {b}, {alpha})"


# CSS-value sanitizer for LLM-supplied theme overrides: colors, gradients and
# lengths only — anything with braces/semicolons/urls is rejected outright.
_SAFE_CSS_VALUE_RE = re.compile(r"^[#a-zA-Z0-9(),.%\s/+-]{1,120}$")


def _safe_css_value(value) -> str:
    v = str(value or "").strip()
    if not v or "url" in v.lower() or not _SAFE_CSS_VALUE_RE.match(v):
        return ""
    return v


def merge_theme_overrides(domain: str, overrides: Dict) -> Dict:
    """Merge LLM-supplied token overrides onto a domain preset's tokens.

    The preset supplies everything not overridden, so a partial override
    ('make it dark green') still yields a complete, coherent theme. Every
    value passes the CSS sanitizer; empty/invalid overrides are ignored.
    """
    t = theme_tokens(domain)
    palette_keys = ("primary", "secondary", "accent", "surface", "background",
                    "text", "muted", "border")
    for key in palette_keys:
        v = _safe_css_value(overrides.get(key))
        if v:
            t["palette"][key] = v
    radius = _safe_css_value(overrides.get("radius"))
    if radius:
        t["radius"] = radius
    hero_bg = _safe_css_value(overrides.get("heroBackground"))
    if hero_bg:
        t["hero_bg"] = hero_bg
    hero_text = _safe_css_value(overrides.get("heroText"))
    if hero_text:
        t["hero_text"] = hero_text
    return t


def stylesheet_rules(domain: str) -> List[Dict]:
    """Return the full GrapesJS ``styles[]`` rule list for ``domain``.

    Includes the base reset, a ``:root`` design-token layer (CSS custom
    properties) and every ``.ds-*`` component class, all derived from the
    domain's tokens. Drop the result straight into ``model["styles"]``.
    """
    return stylesheet_rules_from_tokens(theme_tokens(domain))


def stylesheet_rules_from_tokens(t: Dict) -> List[Dict]:
    """Build the ``.ds-*`` stylesheet from an explicit token bundle.

    Same output as :func:`stylesheet_rules` but for merged/custom tokens
    (see :func:`merge_theme_overrides`) — this is what makes the theme a
    starting point rather than a cage."""
    p = t["palette"]
    sp = t["spacing"]
    ts = t["type_scale"]
    radius = t["radius"]
    shadow = t["shadow"]
    accent_tint = _hex_to_rgba(p["accent"], 0.12)
    btn_primary_bg = t.get("gradient", p["primary"])

    rules: List[Dict] = []

    # --- Base reset + design-token custom properties ---------------------
    rules.append(_raw_rule("*", {"box-sizing": "border-box"}))
    rules.append(_raw_rule(
        ":root",
        {
            "--ds-primary": p["primary"],
            "--ds-secondary": p["secondary"],
            "--ds-accent": p["accent"],
            "--ds-surface": p["surface"],
            "--ds-background": p["background"],
            "--ds-text": p["text"],
            "--ds-muted": p["muted"],
            "--ds-border": p["border"],
            "--ds-radius": radius,
            "--ds-shadow": shadow,
            "--ds-space-md": sp["md"],
            "--ds-space-lg": sp["lg"],
            "--ds-font": t["font"],
            "--ds-font-heading": t["font_heading"],
        },
    ))

    # --- Layout primitives ----------------------------------------------
    rules.append(_class_rule("ds-page", {
        "margin": "0",
        "background-color": p["background"],
        "color": p["text"],
        "font-family": t["font"],
        "font-size": ts["body"]["size"],
        "font-weight": ts["body"]["weight"],
        "line-height": t["line_height"],
        "min-height": "100vh",
    }))
    rules.append(_class_rule("ds-container", {
        "max-width": t["container"],
        "margin": "0 auto",
        "padding-left": sp["md"],
        "padding-right": sp["md"],
        "width": "100%",
    }))
    rules.append(_class_rule("ds-section", {
        "padding-top": sp["section"],
        "padding-bottom": sp["section"],
        "padding-left": sp["md"],
        "padding-right": sp["md"],
    }))

    # --- Navigation ------------------------------------------------------
    rules.append(_class_rule("ds-nav", {
        "display": "flex",
        "align-items": "center",
        "justify-content": "space-between",
        "gap": sp["md"],
        "padding-top": sp["sm"],
        "padding-bottom": sp["sm"],
        "padding-left": sp["md"],
        "padding-right": sp["md"],
        "background-color": p["surface"],
        "border-bottom": f"1px solid {p['border']}",
    }))

    # --- Hero ------------------------------------------------------------
    rules.append(_class_rule("ds-hero", {
        "background": t["hero_bg"],
        "color": t["hero_text"],
        "padding-top": sp["section"],
        "padding-bottom": sp["section"],
        "padding-left": sp["md"],
        "padding-right": sp["md"],
        "text-align": "left",
    }))
    # Headings inside the hero inherit the hero's foreground colour.
    rules.append(_raw_rule(".ds-hero .ds-heading", {"color": t["hero_text"]}))

    # --- Headings --------------------------------------------------------
    rules.append(_class_rule("ds-heading", {
        "margin": f"0 0 {sp['sm']} 0",
        "font-family": t["font_heading"],
        "font-size": ts["h2"]["size"],
        "font-weight": ts["h2"]["weight"],
        "line-height": "1.15",
        "letter-spacing": "-0.02em",
        "color": p["text"],
    }))
    # Hero headings jump to display scale — size contrast is what separates a
    # designed page from a wireframe with colors.
    rules.append(_raw_rule(".ds-hero h1.ds-heading, .ds-hero .ds-heading", {
        "font-size": ts["h1"]["size"],
        "font-weight": ts["h1"]["weight"],
        "letter-spacing": "-0.03em",
        "line-height": "1.08",
        "max-width": "22ch",
    }))

    # --- Card ------------------------------------------------------------
    layered_shadow = (
        f"0 1px 2px {_hex_to_rgba(p['text'], 0.05)}, "
        f"0 4px 16px {_hex_to_rgba(p['text'], 0.06)}"
    )
    raised_shadow = (
        f"0 2px 4px {_hex_to_rgba(p['text'], 0.06)}, "
        f"0 12px 32px {_hex_to_rgba(p['text'], 0.10)}"
    )
    rules.append(_class_rule("ds-card", {
        "background-color": p["surface"],
        "border": f"1px solid {p['border']}",
        "border-radius": radius,
        "padding": sp["lg"],
        "box-shadow": layered_shadow,
        "transition": "box-shadow .2s ease, transform .2s ease",
    }))
    rules.append(_raw_rule(".ds-card:hover", {
        "box-shadow": raised_shadow,
        "transform": "translateY(-1px)",
    }))

    # --- Responsive grids ------------------------------------------------
    rules.append(_class_rule("ds-grid-2", {
        "display": "grid",
        "grid-template-columns": "repeat(2, minmax(0, 1fr))",
        "gap": sp["lg"],
    }))
    rules.append(_class_rule("ds-grid-3", {
        "display": "grid",
        "grid-template-columns": "repeat(3, minmax(0, 1fr))",
        "gap": sp["lg"],
    }))
    rules.append(_media_class_rule(
        "ds-grid-2", "(max-width: 768px)", {"grid-template-columns": "1fr"}))
    rules.append(_media_class_rule(
        "ds-grid-3", "(max-width: 768px)", {"grid-template-columns": "1fr"}))

    # --- KPI / metric card -----------------------------------------------
    rules.append(_class_rule("ds-kpi", {
        "display": "flex",
        "flex-direction": "column",
        "gap": sp["xs"],
        "background-color": p["surface"],
        "border": f"1px solid {p['border']}",
        "border-radius": radius,
        "padding": sp["lg"],
        "box-shadow": layered_shadow,
        "transition": "box-shadow .2s ease, transform .2s ease",
    }))
    rules.append(_raw_rule(".ds-kpi:hover", {
        "box-shadow": raised_shadow,
        "transform": "translateY(-1px)",
    }))
    rules.append(_class_rule("ds-kpi-value", {
        "font-family": t["font_mono"] if t.get("name") == "finance" else t["font_heading"],
        "font-size": ts["h1"]["size"],
        "font-weight": "700",
        "line-height": "1.05",
        "letter-spacing": "-0.02em",
        "color": p["primary"],
    }))
    rules.append(_class_rule("ds-kpi-label", {
        "font-size": ts["small"]["size"],
        "font-weight": "600",
        "letter-spacing": "0.04em",
        "text-transform": "uppercase",
        "color": p["muted"],
    }))

    # --- Table wrapper ---------------------------------------------------
    rules.append(_class_rule("ds-table-wrap", {
        "overflow-x": "auto",
        "background-color": p["surface"],
        "border": f"1px solid {p['border']}",
        "border-radius": radius,
    }))

    # --- Plain data table (class-less standalone GUIs) -------------------
    rules.append(_class_rule("ds-table", {
        "width": "100%",
        "border-collapse": "collapse",
        "background-color": p["surface"],
        "border": f"1px solid {p['border']}",
        "border-radius": radius,
        "overflow": "hidden",
        "font-size": ts["body"]["size"],
        "color": p["text"],
    }))
    rules.append(_raw_rule(".ds-table th", {
        "text-align": "left",
        "padding": "0.75rem 1rem",
        "background-color": p["background"],
        "border-bottom": f"2px solid {p['border']}",
        "font-weight": "600",
        "font-size": "0.8125rem",
        "letter-spacing": "0.02em",
        "text-transform": "uppercase",
        "color": p["muted"],
    }))
    rules.append(_raw_rule(".ds-table td", {
        "padding": "0.75rem 1rem",
        "border-bottom": f"1px solid {p['border']}",
    }))
    rules.append(_raw_rule(".ds-table tbody tr:last-child td", {
        "border-bottom": "none",
    }))

    # --- Buttons ---------------------------------------------------------
    rules.append(_class_rule("ds-btn", {
        "display": "inline-flex",
        "align-items": "center",
        "justify-content": "center",
        "gap": sp["xs"],
        "padding": f"0.625rem 1.25rem",
        "border": f"1px solid {p['border']}",
        "border-radius": radius,
        "background-color": p["surface"],
        "color": p["text"],
        "font-family": t["font"],
        "font-size": ts["body"]["size"],
        "font-weight": "600",
        "text-decoration": "none",
        "cursor": "pointer",
        "transition": "background-color .15s ease, border-color .15s ease, "
                      "filter .15s ease, transform .15s ease",
    }))
    rules.append(_raw_rule(".ds-btn:hover", {
        "border-color": p["muted"],
        "background-color": p["background"],
    }))
    rules.append(_class_rule("ds-btn-primary", {
        "background": btn_primary_bg,
        "border-color": p["primary"],
        "color": "#ffffff",
    }))
    rules.append(_raw_rule(".ds-btn-primary:hover", {
        "filter": "brightness(1.08)",
        "background-color": "transparent",
        "transform": "translateY(-1px)",
    }))

    # --- Notice / alert banner ------------------------------------------
    rules.append(_class_rule("ds-notice", {
        "display": "flex",
        "align-items": "center",
        "gap": sp["sm"],
        "padding": f"{sp['sm']} {sp['md']}",
        "background-color": accent_tint,
        "border-left": f"4px solid {p['accent']}",
        "border-radius": radius,
        "color": p["text"],
        "font-size": ts["body"]["size"],
    }))

    # --- Badge -----------------------------------------------------------
    rules.append(_class_rule("ds-badge", {
        "display": "inline-block",
        "padding": "0.125rem 0.5rem",
        "background-color": accent_tint,
        "border-radius": "999px",
        "color": p["accent"],
        "font-size": ts["small"]["size"],
        "font-weight": "600",
    }))

    # --- Form controls ---------------------------------------------------
    rules.append(_class_rule("ds-field", {
        "display": "flex",
        "flex-direction": "column",
        "gap": sp["xs"],
        "margin-bottom": sp["md"],
    }))
    rules.append(_class_rule("ds-label", {
        "font-size": ts["small"]["size"],
        "font-weight": "600",
        "color": p["text"],
    }))
    rules.append(_class_rule("ds-input", {
        "width": "100%",
        "padding": "0.5rem 0.75rem",
        "background-color": p["surface"],
        "border": f"1px solid {p['border']}",
        "border-radius": radius,
        "color": p["text"],
        "font-family": t["font"],
        "font-size": ts["body"]["size"],
    }))
    rules.append(_raw_rule(".ds-input:focus", {
        "outline": "none",
        "border-color": p["primary"],
        "box-shadow": f"0 0 0 3px {_hex_to_rgba(p['primary'], 0.15)}",
    }))

    # --- Footer ----------------------------------------------------------
    rules.append(_class_rule("ds-footer", {
        "background-color": p["secondary"],
        "color": "#ffffff",
        "padding": f"{sp['xl']} {sp['md']}",
        "font-size": ts["small"]["size"],
    }))

    return rules


# ---------------------------------------------------------------------------
# Block exemplars — proven, editable-safe HTML patterns per domain
# ---------------------------------------------------------------------------
#
# Domain-flavoured copy so Phase 3's prompt shows the LLM realistic patterns.
# Structure is identical across domains (same ``.ds-*`` classes + semantic
# tags); only the sample text/labels change. Data blocks carry a
# ``<!--WIDGET:slot-->`` placeholder that Phase 4 replaces with a live widget.

_CONTENT: Dict[str, Dict] = {
    "government": {
        "hero_title": "Residency Certificate Services",
        "hero_body": "Apply for and track official residency certificates online.",
        "cta": "Start an application",
        "kpis": [("12,480", "Applications processed"),
                 ("3.2 days", "Average wait time"),
                 ("28", "Service offices open")],
        "table_title": "Recent applications",
        "chart_title": "Applications by district",
        "notice": "Scheduled maintenance on Sunday 02:00-04:00 CET.",
        "form_title": "Request a certificate",
        "fields": [("Full legal name", "text"),
                   ("National ID number", "text"),
                   ("Purpose of request", "text")],
        "footer": "Ministry of Civic Services",
    },
    "finance": {
        "hero_title": "Portfolio Overview",
        "hero_body": "Track positions, performance and cash flow in real time.",
        "cta": "Open dashboard",
        "kpis": [("$1.24M", "Total assets"),
                 ("+2.8%", "Daily P&L"),
                 ("47", "Open positions")],
        "table_title": "Recent transactions",
        "chart_title": "Revenue by quarter",
        "notice": "Markets close today at 16:00 EST.",
        "form_title": "New trade order",
        "fields": [("Ticker symbol", "text"),
                   ("Quantity", "number"),
                   ("Order type", "text")],
        "footer": "Meridian Capital",
    },
    "health": {
        "hero_title": "Your Patient Portal",
        "hero_body": "Book appointments, message your care team and view results.",
        "cta": "Book an appointment",
        "kpis": [("2", "Upcoming appointments"),
                 ("5", "Unread messages"),
                 ("3", "Active prescriptions")],
        "table_title": "Appointment history",
        "chart_title": "Vitals over time",
        "notice": "Your next appointment is Thursday at 10:30 AM.",
        "form_title": "Book an appointment",
        "fields": [("Full name", "text"),
                   ("Reason for visit", "text"),
                   ("Preferred date", "date")],
        "footer": "Lakeside Health Network",
    },
    "startup": {
        "hero_title": "Ship products your users love",
        "hero_body": "The all-in-one workspace to plan, build and launch faster.",
        "cta": "Get early access",
        "kpis": [("18.2K", "Active users"),
                 ("$42K", "Monthly recurring revenue"),
                 ("+31%", "Month-over-month growth")],
        "table_title": "Recent signups",
        "chart_title": "Weekly active users",
        "notice": "New: launch-week deals are live for 7 days only.",
        "form_title": "Get early access",
        "fields": [("Work email", "email"),
                   ("Company", "text"),
                   ("Team size", "number")],
        "footer": "Nova Labs",
    },
    "default": {
        "hero_title": "Welcome to your dashboard",
        "hero_body": "A clean starting point to view and manage your data.",
        "cta": "Get started",
        "kpis": [("1,204", "Total items"),
                 ("312", "Active"),
                 ("18", "Pending")],
        "table_title": "Recent records",
        "chart_title": "Overview",
        "notice": "Tip: use the sidebar to switch between views.",
        "form_title": "Add new record",
        "fields": [("Name", "text"),
                   ("Category", "text"),
                   ("Notes", "text")],
        "footer": "Your Company",
    },
}

# Small inline SVG info glyph for the notice banner (inline SVG is CSP-safe).
_INFO_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10">'
    '</circle><line x1="12" y1="16" x2="12" y2="12"></line>'
    '<line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
)


def block_exemplars(domain: str) -> Dict[str, str]:
    """Return domain-appropriate HTML block snippets keyed by block type.

    Keys: ``hero``, ``kpi_row``, ``table_card``, ``chart_card``, ``footer``,
    ``form``, ``notice``. All markup uses the ``.ds-*`` component classes plus
    semantic text tags; the ``table_card`` and ``chart_card`` data blocks embed
    a ``<!--WIDGET:slot-->`` placeholder for Phase 4. No external images or
    webfonts — imagery is CSS-driven or inline SVG only.
    """
    c = _CONTENT.get(domain) or _CONTENT[DEFAULT_DOMAIN]

    hero = (
        '<section class="ds-hero">\n'
        '  <div class="ds-container">\n'
        f'    <h1 class="ds-heading">{c["hero_title"]}</h1>\n'
        f'    <p>{c["hero_body"]}</p>\n'
        f'    <a class="ds-btn ds-btn-primary" href="#">{c["cta"]}</a>\n'
        '  </div>\n'
        '</section>'
    )

    kpi_cards = "\n".join(
        '    <div class="ds-kpi">\n'
        f'      <span class="ds-kpi-value">{value}</span>\n'
        f'      <span class="ds-kpi-label">{label}</span>\n'
        '    </div>'
        for value, label in c["kpis"]
    )
    kpi_row = (
        '<section class="ds-section">\n'
        '  <div class="ds-grid-3">\n'
        f'{kpi_cards}\n'
        '  </div>\n'
        '</section>'
    )

    table_card = (
        '<section class="ds-section">\n'
        '  <div class="ds-card">\n'
        f'    <h3 class="ds-heading">{c["table_title"]}</h3>\n'
        '    <div class="ds-table-wrap">\n'
        '      <!--WIDGET:table-->\n'
        '    </div>\n'
        '  </div>\n'
        '</section>'
    )

    chart_card = (
        '<section class="ds-section">\n'
        '  <div class="ds-card">\n'
        f'    <h3 class="ds-heading">{c["chart_title"]}</h3>\n'
        '    <div class="ds-chart">\n'
        '      <!--WIDGET:chart-->\n'
        '    </div>\n'
        '  </div>\n'
        '</section>'
    )

    footer = (
        '<footer class="ds-footer">\n'
        '  <div class="ds-container">\n'
        f'    <p>&copy; {c["footer"]}. All rights reserved.</p>\n'
        '  </div>\n'
        '</footer>'
    )

    field_html = "\n".join(
        '    <div class="ds-field">\n'
        f'      <label class="ds-label">{label}</label>\n'
        f'      <input class="ds-input" type="{ftype}" />\n'
        '    </div>'
        for label, ftype in c["fields"]
    )
    form = (
        '<section class="ds-section">\n'
        '  <form class="ds-card">\n'
        f'    <h3 class="ds-heading">{c["form_title"]}</h3>\n'
        f'{field_html}\n'
        f'    <button class="ds-btn ds-btn-primary" type="submit">{c["cta"]}</button>\n'
        '  </form>\n'
        '</section>'
    )

    notice = (
        '<div class="ds-notice">\n'
        f'  {_INFO_SVG}\n'
        f'  <span>{c["notice"]}</span>\n'
        '</div>'
    )

    return {
        "hero": hero,
        "kpi_row": kpi_row,
        "table_card": table_card,
        "chart_card": chart_card,
        "footer": footer,
        "form": form,
        "notice": notice,
    }
