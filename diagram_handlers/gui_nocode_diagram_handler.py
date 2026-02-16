"""
GUI No-Code Diagram Handler
Handles generation of GUINoCodeDiagram models for GrapesJS-based editor.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

from .base_handler import BaseDiagramHandler

DEFAULT_GUI_VERSION = "0.21.13"


def _clean_text(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else fallback
    return fallback


def _sanitize_page_name(value: Any, fallback: str = "Page") -> str:
    label = _clean_text(value, fallback=fallback)
    if not label:
        return fallback
    label = re.sub(r"\s+", " ", label)
    return label[:40]


def _default_wrapper_component() -> Dict[str, Any]:
    return {
        "type": "wrapper",
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


def _hero_component(title: str, body: str, cta_label: str) -> Dict[str, Any]:
    return {
        "tagName": "section",
        "attributes": {"class": "assistant-hero"},
        "style": {
            "padding": "48px 32px",
            "background": "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            "color": "#ffffff",
            "border-radius": "18px",
            "margin": "20px 0",
        },
        "components": [
            {
                "tagName": "h1",
                "content": title,
                "style": {"margin": "0 0 12px 0", "font-size": "2rem", "font-weight": "700"},
            },
            {
                "tagName": "p",
                "content": body,
                "style": {"margin": "0 0 20px 0", "font-size": "1.05rem", "line-height": "1.5"},
            },
            {
                "tagName": "button",
                "content": cta_label,
                "attributes": {"class": "assistant-cta"},
                "style": {
                    "padding": "10px 18px",
                    "border": "none",
                    "border-radius": "10px",
                    "font-weight": "600",
                    "background-color": "#38bdf8",
                    "color": "#0f172a",
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
        "style": {"padding": "28px", "background-color": "#f8fafc", "border-radius": "14px", "margin": "16px 0"},
        "components": [
            {
                "tagName": "h2",
                "content": title,
                "style": {"margin": "0 0 14px 0", "font-size": "1.45rem"},
            },
            {
                "tagName": "ul",
                "style": {"padding-left": "20px", "margin": "0"},
                "components": [{"tagName": "li", "content": item, "style": {"margin": "8px 0"}} for item in cleaned_items],
            },
        ],
    }


def _content_component(title: str, body: str) -> Dict[str, Any]:
    return {
        "tagName": "section",
        "attributes": {"class": "assistant-content"},
        "style": {"padding": "24px", "border": "1px solid #e2e8f0", "border-radius": "12px", "margin": "14px 0"},
        "components": [
            {"tagName": "h2", "content": title, "style": {"margin": "0 0 10px 0", "font-size": "1.35rem"}},
            {"tagName": "p", "content": body, "style": {"margin": "0", "line-height": "1.55"}},
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
            "padding": "26px",
            "border": "1px solid #cbd5e1",
            "border-radius": "12px",
            "margin": "16px 0",
            "background-color": "#ffffff",
        },
        "components": [
            {"tagName": "h2", "content": title, "style": {"margin": "0 0 14px 0", "font-size": "1.3rem"}},
            {
                "tagName": "form",
                "components": [
                    {
                        "tagName": "div",
                        "style": {"display": "grid", "gap": "10px"},
                        "components": [
                            {
                                "tagName": "input",
                                "attributes": {
                                    "type": "text",
                                    "name": re.sub(r"[^a-z0-9_]+", "_", field.lower()),
                                    "placeholder": field,
                                },
                                "style": {
                                    "padding": "10px 12px",
                                    "border": "1px solid #94a3b8",
                                    "border-radius": "8px",
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
                            "margin-top": "12px",
                            "padding": "10px 16px",
                            "border": "none",
                            "border-radius": "8px",
                            "background-color": "#0f172a",
                            "color": "#ffffff",
                            "font-weight": "600",
                        },
                    },
                ],
            },
        ],
    }


def _build_section_component(section_spec: Dict[str, Any]) -> Dict[str, Any]:
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
    return _content_component(title, body)


class GUINoCodeDiagramHandler(BaseDiagramHandler):
    """Handler for GUI no-code diagram generation."""

    def get_diagram_type(self) -> str:
        return "GUINoCodeDiagram"

    def get_system_prompt(self) -> str:
        return """You are a UI modeling expert for a no-code web editor.

Return ONLY JSON with this shape:
{
  "pageName": "Home",
  "section": {
    "type": "hero|feature_list|content|form",
    "title": "Section title",
    "body": "Optional descriptive text",
    "items": ["Optional item"],
    "fields": ["Optional field label"],
    "ctaLabel": "Optional button label"
  }
}

Rules:
1. Keep content concise and practical.
2. Use section type that best matches user request.
3. Return JSON only."""

    def _parse_page_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        page_name = _sanitize_page_name(spec.get("name"), fallback="Page")
        raw_sections = spec.get("sections") if isinstance(spec.get("sections"), list) else []
        sections = [item for item in raw_sections if isinstance(item, dict)]

        wrapper = _default_wrapper_component()
        wrapper["components"] = [_build_section_component(section) for section in sections]

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

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        prompt = self.get_system_prompt()

        try:
            response = self.llm.predict(f"{prompt}\n\nUser Request: {user_request}")
            spec = self.parse_json_safely(self.clean_json_response(response or ""))
            if not isinstance(spec, dict):
                raise ValueError("Invalid section spec")

            page_name = _sanitize_page_name(spec.get("pageName"), fallback="Home")
            section_spec = spec.get("section") if isinstance(spec.get("section"), dict) else {}
            section_component = _build_section_component(section_spec)

            model = _default_gui_model()
            model = self._append_section(model, page_name, section_component)
            return {
                "action": "inject_element",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": f"Added a new UI section to page '{page_name}'.",
            }
        except Exception:
            return self.generate_fallback_element(user_request)

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None) -> Dict[str, Any]:
        system_prompt = """You are a UI modeling expert.

Return ONLY JSON with this shape:
{
  "projectName": "Name",
  "pages": [
    {
      "name": "Home",
      "sections": [
        {
          "type": "hero|feature_list|content|form",
          "title": "Section title",
          "body": "Optional text",
          "items": ["Optional item"],
          "fields": ["Optional field"],
          "ctaLabel": "Optional CTA"
        }
      ]
    }
  ]
}

Rules:
1. Create 1-4 pages depending on request complexity.
2. Each page should include 1-5 sections.
3. Return JSON only."""

        try:
            response = self.llm.predict(f"{system_prompt}\n\nUser Request: {user_request}")
            spec = self.parse_json_safely(self.clean_json_response(response or ""))
            if not isinstance(spec, dict):
                raise ValueError("Invalid system spec")

            pages_spec = spec.get("pages") if isinstance(spec.get("pages"), list) else []
            pages = [self._parse_page_spec(page) for page in pages_spec if isinstance(page, dict)]
            if not pages:
                fallback = self.generate_fallback_system()
                return fallback

            model = {
                "pages": pages,
                "styles": [],
                "assets": [],
                "symbols": [],
                "version": DEFAULT_GUI_VERSION,
            }

            return {
                "action": "inject_complete_system",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": f"Created GUI model with {len(pages)} page(s).",
            }
        except Exception:
            return self.generate_fallback_system()

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None) -> Dict[str, Any]:
        model = _normalize_gui_model(current_model)
        page_names = [
            _clean_text(page.get("name"))
            for page in model.get("pages", [])
            if isinstance(page, dict) and _clean_text(page.get("name"))
        ]
        pages_hint = ", ".join(page_names) if page_names else "Home"

        prompt = """You are a UI modeling assistant.

Return ONLY JSON with this shape:
{
  "operation": "append_section|rename_page|remove_page",
  "pageName": "Target page",
  "newPageName": "Required for rename_page",
  "section": {
    "type": "hero|feature_list|content|form",
    "title": "Section title",
    "body": "Optional text",
    "items": ["Optional item"],
    "fields": ["Optional field"],
    "ctaLabel": "Optional CTA"
  }
}

Rules:
1. Prefer append_section when request asks to add/update content.
2. Use existing page names when possible.
3. Return JSON only."""

        try:
            response = self.llm.predict(
                f"{prompt}\n\nAvailable pages: {pages_hint}\n\nUser Request: {user_request}"
            )
            spec = self.parse_json_safely(self.clean_json_response(response or ""))
            if not isinstance(spec, dict):
                raise ValueError("Invalid modification spec")

            operation = _clean_text(spec.get("operation"), fallback="append_section")
            page_name = _sanitize_page_name(spec.get("pageName"), fallback=page_names[0] if page_names else "Home")

            if operation == "rename_page":
                new_page_name = _sanitize_page_name(spec.get("newPageName"), fallback=page_name)
                for page in model.get("pages", []):
                    if not isinstance(page, dict):
                        continue
                    if _clean_text(page.get("name")).lower() == page_name.lower():
                        page["name"] = new_page_name
                        page["route_path"] = f"/{re.sub(r'[^a-z0-9-]+', '-', new_page_name.lower()).strip('-') or 'page'}"
                        break
                message = f"Renamed page '{page_name}' to '{new_page_name}'."
            elif operation == "remove_page":
                filtered_pages = [
                    page
                    for page in model.get("pages", [])
                    if not isinstance(page, dict) or _clean_text(page.get("name")).lower() != page_name.lower()
                ]
                if filtered_pages:
                    model["pages"] = filtered_pages
                message = f"Removed page '{page_name}' from the GUI model."
            else:
                section_spec = spec.get("section") if isinstance(spec.get("section"), dict) else {}
                section_component = _build_section_component(section_spec)
                model = self._append_section(model, page_name, section_component)
                message = f"Added a new section to page '{page_name}'."

            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": message,
            }
        except Exception:
            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": "Could not parse the requested GUI modification, but kept the existing model safe.",
            }

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
            "message": "Created a basic GUI section (fallback).",
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
            "message": "Created a basic GUI model (fallback).",
        }
