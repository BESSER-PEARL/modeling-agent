"""LLM-authored HTML → GrapesJS component-definition tree converter.

Phase 1 of the no-code GUI rebuild. This module turns rich themed markup
written by the LLM into the nested component-definition nodes the GUI editor
already loads via ``loadProjectData`` — the same node shape the typed Python
builders in ``gui_nocode_diagram_handler.py`` emit (e.g. ``_hero_component``).

Node shapes (matching the existing builders exactly):

* element node — ``{"tagName": <tag>, "attributes": {...}, "style": {...},
  "components": [<child nodes>]}``. An element whose only child is text carries
  its text via ``content`` (like the current builders) instead of a textnode
  child; mixed content uses textnode children.
* text node — ``{"type": "textnode", "content": <text>}``.

Design constraints baked in here:

* **Tag identity is preserved** — an ``<h2>`` becomes ``tagName:"h2"``, never
  collapsed to a div. The frontend's ``markTextEditable`` coerces text tags
  (``p``/``h1``-``h6``/``span``/``a``/``li``/``label``/``blockquote``) to
  double-click-editable text purely by tag name, so we must NOT stamp
  ``type:"text"`` ourselves — that is the frontend's job and stamping it here
  would break editability.
* **Sanitised** — ``<script>``/``<style>`` dropped, ``on*`` handlers stripped,
  external ``href``/``src`` values dropped (CSP blocks them), only ``data:``
  URIs and ``#`` anchors survive.
* **Widget-spoof guarded** — LLM markup can never masquerade as a data-bound
  widget: ``data-gjs-type``/``data-source``/``series``/``columns`` attributes
  are dropped and no parsed node is ever given a top-level ``type`` of a widget.
  Real widgets come only from the typed Python builders (a later phase).

The module is pure/standalone — it never imports the handler.
"""

from __future__ import annotations

import copy
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Whitelists
# ---------------------------------------------------------------------------

#: Structural / text / media tags safe to keep verbatim, plus inline-SVG icon
#: tags and plain (NON data-bound) table tags.
SAFE_TAGS = frozenset({
    # layout / structure
    "div", "section", "header", "footer", "nav", "main", "article", "aside",
    # headings
    "h1", "h2", "h3", "h4", "h5", "h6",
    # text / inline
    "p", "span", "a", "ul", "ol", "li", "img", "button", "label",
    "strong", "em", "b", "i", "small", "br", "hr", "blockquote",
    "figure", "figcaption",
    # form controls (static mockup fields — GrapesJS renders them editable)
    "form", "fieldset", "legend", "input", "textarea", "select", "option",
    "optgroup",
    # inline svg (icons)
    "svg", "path", "g", "circle", "rect", "line", "polyline", "polygon", "text",
    # plain, non data-bound tables
    "table", "thead", "tbody", "tr", "td", "th",
})

#: Tags that never have children — treated as self-closing so the parser's
#: lack of HTML void-element knowledge doesn't leave them "open".
VOID_TAGS = frozenset({
    "img", "br", "hr", "input",
    "path", "circle", "rect", "line", "polyline", "polygon",
})

#: Content of these tags is dropped entirely (not flattened).
DROP_TAGS = frozenset({"script", "style"})

#: Attributes that would let LLM markup spoof a real data-bound widget, or that
#: the frontend uses to assign a GrapesJS component type. Always dropped.
_WIDGET_ATTR_BLOCKLIST = frozenset({
    "data-gjs-type", "data-source", "series", "columns", "data-widget-slot",
})

#: html.parser lowercases attribute names; restore the few camelCase SVG
#: attributes that are case-sensitive so icons still render correctly.
_SVG_ATTR_CASE = {
    "viewbox": "viewBox",
    "preserveaspectratio": "preserveAspectRatio",
}

#: Node ``type`` used for a widget splice point the caller fills in later.
WIDGET_SLOT_TYPE = "widget-slot"

_WIDGET_COMMENT_RE = re.compile(r"^\s*WIDGET:\s*(.+?)\s*$")


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _is_safe_url(value: str) -> bool:
    """A ``href``/``src`` value survives only if it is a data: URI or #anchor.

    Everything else (http(s), protocol-relative, ``javascript:``, bare paths)
    is an external asset the same-origin CSP blocks, so it is dropped.
    """
    v = (value or "").strip()
    if not v:
        return False
    return v.lower().startswith("data:") or v.startswith("#")


def _split_declarations(style_str: str) -> List[str]:
    """Split an inline style string on ``;`` while respecting parentheses.

    Keeps ``url(data:image/svg+xml;base64,...)`` intact — a naive ``split(';')``
    would corrupt data: URIs that carry a ``;base64`` marker.
    """
    decls: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in style_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == ";" and depth == 0:
            decls.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        decls.append("".join(current))
    return decls


def _parse_style(style_str: str) -> Dict[str, str]:
    """Parse an inline ``style="prop: val; prop2: val2"`` into a css dict.

    GrapesJS wants ``style`` as a mapping of css-property -> value, mirroring
    the ``style`` dicts the typed builders emit.
    """
    style: Dict[str, str] = {}
    for decl in _split_declarations(style_str or ""):
        if ":" not in decl:
            continue
        prop, _, val = decl.partition(":")  # first colon only (keeps url(http:))
        prop = prop.strip().lower()
        val = val.strip()
        if prop and val:
            style[prop] = val
    return style


def _widget_slot_node(slot_id: str) -> Dict[str, Any]:
    """A locatable marker node the caller later swaps for a real typed widget."""
    return {"type": WIDGET_SLOT_TYPE, "attributes": {"data-widget-slot": slot_id}}


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

class _Frame:
    """A single open element while parsing.

    ``node`` is the element dict being built (``None`` for the document root or
    for a dropped/flattened tag). ``children`` is the list child nodes are
    appended to — for a flattened tag it is the *parent's* list, so the tag's
    safe children survive as siblings.
    """

    __slots__ = ("tag", "node", "children")

    def __init__(self, tag: Optional[str], node: Optional[Dict[str, Any]],
                 children: List[Dict[str, Any]]):
        self.tag = tag
        self.node = node
        self.children = children


class _ComponentTreeBuilder(HTMLParser):
    """Streams HTML into a nested list of GrapesJS component-definition nodes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._root: List[Dict[str, Any]] = []
        self._stack: List[_Frame] = [_Frame(None, None, self._root)]
        self._drop_data = False  # inside <script>/<style> CDATA

    # -- public result -----------------------------------------------------

    def result(self) -> List[Dict[str, Any]]:
        """Finalise any unclosed tags and return the root component list."""
        while len(self._stack) > 1:
            self._pop()
        return self._root

    # -- element attribute → node ------------------------------------------

    def _build_node(self, tag: str, attrs: List[tuple]) -> Dict[str, Any]:
        """Turn a whitelisted tag + its attributes into a bare element node.

        Sanitises as it goes: ``on*`` handlers dropped, ``style`` parsed into a
        dict, ``href``/``src`` kept only when safe, widget-spoofing attributes
        dropped. ``class``/``id``/``data-*`` and other benign attributes (svg
        geometry, ``alt``, ``type``, ...) are preserved.
        """
        attributes: Dict[str, Any] = {}
        style: Dict[str, str] = {}

        for raw_name, raw_value in attrs:
            name = (raw_name or "").lower()
            value = raw_value if raw_value is not None else ""

            if name.startswith("on"):
                continue  # strip event handlers (onclick, onerror, ...)
            if name in _WIDGET_ATTR_BLOCKLIST:
                continue  # never let markup spoof a data-bound widget
            if name == "style":
                style.update(_parse_style(value))
                continue
            if name in ("href", "src"):
                if _is_safe_url(value):
                    attributes[name] = value
                continue  # external asset → drop the attribute, keep the element
            if name in ("class", "id"):
                if value:
                    attributes[name] = value
                continue
            attributes[_SVG_ATTR_CASE.get(name, name)] = value

        node: Dict[str, Any] = {"tagName": tag}
        if attributes:
            node["attributes"] = attributes
        if style:
            node["style"] = style
        return node

    # -- HTMLParser hooks --------------------------------------------------

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        self._open(tag, attrs, self_closing=False)

    def handle_startendtag(self, tag: str, attrs: List[tuple]) -> None:
        self._open(tag, attrs, self_closing=True)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in DROP_TAGS:
            self._drop_data = False
            return
        # Pop back to the matching open frame (tolerates missing/misnested ends).
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                while len(self._stack) > i:
                    self._pop()
                return
        # No matching open tag — ignore the stray end tag.

    def handle_data(self, data: str) -> None:
        if self._drop_data or not data:
            return
        collapsed = re.sub(r"\s+", " ", data)
        if not collapsed.strip():
            return  # pure formatting whitespace between tags
        self._stack[-1].children.append({"type": "textnode", "content": collapsed})

    def handle_comment(self, data: str) -> None:
        if self._drop_data:
            return
        match = _WIDGET_COMMENT_RE.match(data or "")
        if match:
            slot_id = match.group(1).strip()
            if slot_id:
                self._stack[-1].children.append(_widget_slot_node(slot_id))

    # -- open / close mechanics --------------------------------------------

    def _open(self, tag: str, attrs: List[tuple], self_closing: bool) -> None:
        tag = tag.lower()
        if self._drop_data:
            return
        if tag in DROP_TAGS:
            if not self_closing:
                self._drop_data = True  # enter CDATA-drop until the end tag
            return

        parent = self._stack[-1]

        # Explicit widget splice point: <div data-widget-slot="slot_id"></div>.
        attr_map = {(k or "").lower(): (v or "") for k, v in attrs}
        if tag == "div" and "data-widget-slot" in attr_map:
            slot_id = attr_map["data-widget-slot"].strip()
            if slot_id:
                parent.children.append(_widget_slot_node(slot_id))
                if not self_closing:
                    # Flatten any (unexpected) children out as siblings.
                    self._stack.append(_Frame(tag, None, parent.children))
                return

        if tag in SAFE_TAGS:
            node = self._build_node(tag, attrs)
            parent.children.append(node)
            if not self_closing and tag not in VOID_TAGS:
                self._stack.append(_Frame(tag, node, []))
        elif not self_closing:
            # Unknown/unsafe tag: drop the tag but keep + flatten safe children.
            self._stack.append(_Frame(tag, None, parent.children))

    def _pop(self) -> None:
        """Close the top frame, attaching its collected children to its node."""
        frame = self._stack.pop()
        node = frame.node
        if node is None:  # root or flattened tag — children already placed
            return
        children = frame.children
        if not children:
            return
        # Single text child → ``content`` (matches the typed builders); mixed or
        # element content → textnode/element children under ``components``.
        if len(children) == 1 and children[0].get("type") == "textnode":
            node["content"] = children[0]["content"]
        else:
            node["components"] = children


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def html_to_components(html: str) -> List[Dict[str, Any]]:
    """Convert LLM-authored HTML into a GrapesJS component-definition tree.

    Returns a list of top-level component nodes ready to drop into a frame's
    ``components`` (the same shape ``loadProjectData`` consumes). Malformed or
    empty input yields an empty list rather than raising.
    """
    if not isinstance(html, str) or not html.strip():
        return []
    builder = _ComponentTreeBuilder()
    builder.feed(html)
    builder.close()
    return builder.result()


def find_widget_slots(nodes: List[Dict[str, Any]]) -> List[str]:
    """Return the slot ids of every widget-slot marker in *nodes* (depth-first)."""
    found: List[str] = []

    def _walk(items: Any) -> None:
        if not isinstance(items, list):
            return
        for node in items:
            if not isinstance(node, dict):
                continue
            if node.get("type") == WIDGET_SLOT_TYPE:
                slot_id = node.get("attributes", {}).get("data-widget-slot")
                if slot_id:
                    found.append(slot_id)
            _walk(node.get("components"))

    _walk(nodes)
    return found


def replace_widget_slot(
    nodes: List[Dict[str, Any]],
    slot_id: str,
    widget_node: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return a NEW tree with every ``slot_id`` marker replaced by *widget_node*.

    Deep — recurses into ``components`` — and non-mutating: the input tree and
    *widget_node* are left untouched (a fresh deepcopy of the widget is spliced
    in at each matching slot).
    """
    def _transform(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in items:
            if (
                isinstance(node, dict)
                and node.get("type") == WIDGET_SLOT_TYPE
                and node.get("attributes", {}).get("data-widget-slot") == slot_id
            ):
                out.append(copy.deepcopy(widget_node))
                continue
            new_node = copy.deepcopy(node)
            if isinstance(new_node, dict) and isinstance(new_node.get("components"), list):
                new_node["components"] = _transform(new_node["components"])
            out.append(new_node)
        return out

    return _transform(nodes)
