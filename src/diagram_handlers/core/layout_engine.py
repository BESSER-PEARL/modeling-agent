"""
Deterministic Layout Engine for UML Diagrams.

Computes element positions algorithmically so the LLM never needs to
generate pixel coordinates.  Supports Class, Object, StateMachine, and
Agent diagrams with collision-avoidance against existing canvas elements.

Design principles:
 - The LLM produces *semantic* output only (names, attributes, relationships …).
 - After parsing, the layout engine assigns ``position`` to every element.
 - Existing elements (from the current model) are treated as occupied
   rectangles that new elements must avoid.
 - Relationship / transition edges influence grouping (connected elements
   are placed near each other).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Canvas & sizing constants (match the web editor coordinate system)
# ---------------------------------------------------------------------------

_BASE_CANVAS_MIN_X = -900
_BASE_CANVAS_MAX_X = 900
_BASE_CANVAS_MIN_Y = -500
_BASE_CANVAS_MAX_Y = 500

# Active canvas bounds — may be expanded dynamically for large diagrams
CANVAS_MIN_X = _BASE_CANVAS_MIN_X
CANVAS_MAX_X = _BASE_CANVAS_MAX_X
CANVAS_MIN_Y = _BASE_CANVAS_MIN_Y
CANVAS_MAX_Y = _BASE_CANVAS_MAX_Y

# Element sizing defaults per diagram type
CLASS_WIDTH = 220
CLASS_HEADER_HEIGHT = 50
CLASS_ATTR_ROW = 25
CLASS_METHOD_ROW = 25
CLASS_MIN_HEIGHT = 90

OBJECT_WIDTH = 220
OBJECT_HEADER_HEIGHT = 50
OBJECT_ATTR_ROW = 25
OBJECT_MIN_HEIGHT = 90

STATE_WIDTH = 220
STATE_MIN_HEIGHT = 80
STATE_ACTION_ROW = 20

AGENT_STATE_WIDTH = 210
AGENT_INTENT_WIDTH = 230
AGENT_NODE_MIN_HEIGHT = 80
AGENT_REPLY_ROW = 20
AGENT_PHRASE_ROW = 18
INITIAL_NODE_SIZE = 45

# Spacing & margins
H_GAP = 100         # horizontal gap between elements
V_GAP = 80          # vertical gap between elements
REL_EXTRA_GAP = 60  # additional gap between classes connected by a relationship
MARGIN = 40         # minimum margin from any existing occupied rect
GRID_SNAP = 20      # snap coordinates to multiples of this value


def _dynamic_canvas_bounds(num_elements: int) -> tuple:
    """Return (min_x, max_x, min_y, max_y) expanded for large diagrams.

    The web editor canvas can scroll arbitrarily, so we expand beyond
    the default viewport for big diagrams to avoid cramming elements.
    """
    if num_elements <= 6:
        return _BASE_CANVAS_MIN_X, _BASE_CANVAS_MAX_X, _BASE_CANVAS_MIN_Y, _BASE_CANVAS_MAX_Y

    # Scale canvas proportionally: +300px per 4 extra elements
    extra = num_elements - 6
    expand_x = (extra // 4 + 1) * 300
    expand_y = (extra // 4 + 1) * 200
    return (
        _BASE_CANVAS_MIN_X - expand_x,
        _BASE_CANVAS_MAX_X + expand_x,
        _BASE_CANVAS_MIN_Y - expand_y,
        _BASE_CANVAS_MAX_Y + expand_y,
    )


def _ideal_grid_shape(n: int) -> tuple:
    """Return (rows, cols) for an approximately square grid of n elements.

    Prefers slightly wider than taller layouts (cols >= rows).
    """
    import math
    if n <= 0:
        return (1, 1)
    cols = math.ceil(math.sqrt(n * 1.4))  # bias toward wider
    rows = math.ceil(n / cols)
    return (rows, cols)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class Rect:
    """Axis-aligned bounding rectangle."""
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def expanded(self, margin: int) -> "Rect":
        return Rect(self.x - margin, self.y - margin,
                     self.width + 2 * margin, self.height + 2 * margin)

    def overlaps(self, other: "Rect") -> bool:
        return not (self.right <= other.x or other.right <= self.x or
                    self.bottom <= other.y or other.bottom <= self.y)


def _snap(value: int) -> int:
    """Snap a coordinate value to the nearest grid multiple."""
    return round(value / GRID_SNAP) * GRID_SNAP


@dataclass
class LayoutItem:
    """An element waiting to be positioned."""
    key: str                          # logical name / id used for edge references
    width: int
    height: int
    group: str = ""                   # optional grouping hint (e.g. "parent", "child")
    order_hint: int = 0               # optional ordering priority (lower = placed first)
    position: Optional[Tuple[int, int]] = None  # assigned by layout engine


# ---------------------------------------------------------------------------
# Size estimators
# ---------------------------------------------------------------------------

def estimate_class_size(spec: Dict[str, Any]) -> Tuple[int, int]:
    """Return (width, height) for a class element spec."""
    n_attrs = len(spec.get("attributes", []))
    n_methods = len(spec.get("methods", []))
    height = CLASS_HEADER_HEIGHT + n_attrs * CLASS_ATTR_ROW + n_methods * CLASS_METHOD_ROW
    return CLASS_WIDTH, max(height, CLASS_MIN_HEIGHT)


def estimate_object_size(spec: Dict[str, Any]) -> Tuple[int, int]:
    """Return (width, height) for an object element spec."""
    n_attrs = len(spec.get("attributes", []))
    height = OBJECT_HEADER_HEIGHT + n_attrs * OBJECT_ATTR_ROW
    return OBJECT_WIDTH, max(height, OBJECT_MIN_HEIGHT)


def estimate_state_size(spec: Dict[str, Any]) -> Tuple[int, int]:
    """Return (width, height) for a state element spec."""
    extra_rows = 0
    if spec.get("entryAction"):
        extra_rows += 1
    if spec.get("exitAction"):
        extra_rows += 1
    if spec.get("doActivity"):
        extra_rows += 1
    height = STATE_MIN_HEIGHT + extra_rows * STATE_ACTION_ROW
    stype = spec.get("stateType", "regular")
    if stype in ("initial", "final"):
        return INITIAL_NODE_SIZE, INITIAL_NODE_SIZE
    return STATE_WIDTH, max(height, STATE_MIN_HEIGHT)


def _estimate_text_width(texts: List[Any], base_width: int, char_px: int = 11, padding: int = 40) -> int:
    """Estimate rendered width from the longest text line.

    The frontend auto-sizes elements based on content.  This heuristic
    uses ~char_px pixels per character + padding to approximate the
    rendered width so the layout engine can avoid overlaps.
    """
    max_w = base_width
    for text in texts:
        if isinstance(text, str) and text:
            estimated = len(text) * char_px + padding
            max_w = max(max_w, estimated)
    return max_w


def estimate_agent_element_size(spec: Dict[str, Any]) -> Tuple[int, int]:
    """Return (width, height) for an agent diagram element spec."""
    elem_type = spec.get("type", "state")
    if elem_type == "initial":
        return INITIAL_NODE_SIZE, INITIAL_NODE_SIZE
    if elem_type == "intent":
        phrases = spec.get("trainingPhrases", [])
        n_phrases = len(phrases)
        height = AGENT_NODE_MIN_HEIGHT + n_phrases * AGENT_PHRASE_ROW
        # Estimate width from longest training phrase
        width = _estimate_text_width(phrases, AGENT_INTENT_WIDTH)
        return width, max(height, AGENT_NODE_MIN_HEIGHT)
    # state (default)
    n_replies = len(spec.get("replies", []))
    n_fallback = len(spec.get("fallbackBodies", []))
    height = AGENT_NODE_MIN_HEIGHT + (n_replies + n_fallback) * AGENT_REPLY_ROW
    # Estimate width from longest reply or fallback text
    all_texts = list(spec.get("replies", [])) + list(spec.get("fallbackBodies", []))
    width = _estimate_text_width(all_texts, AGENT_STATE_WIDTH)
    return width, max(height, AGENT_NODE_MIN_HEIGHT)


# ---------------------------------------------------------------------------
# Occupied-area extraction from an existing model
# ---------------------------------------------------------------------------

_PRIMARY_ELEMENT_TYPES: Dict[str, Set[str]] = {
    "ClassDiagram": {"Class"},
    "ObjectDiagram": {"Object"},
    "StateMachineDiagram": {"State", "StateInitialNode", "StateFinalNode"},
    "AgentDiagram": {"AgentState", "AgentIntent", "StateInitialNode"},
}

_CHILD_ELEMENT_TYPES: Set[str] = {
    "ClassAttribute", "ClassMethod",
    "AgentStateBody", "AgentStateFallbackBody", "AgentIntentBody",
}


def extract_occupied_rects(
    model: Optional[Dict[str, Any]],
    diagram_type: str,
) -> List[Rect]:
    """Parse the existing model and return occupied rectangles for primary elements."""
    if not isinstance(model, dict):
        return []
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return []

    primary_types = _PRIMARY_ELEMENT_TYPES.get(diagram_type, set())
    rects: List[Rect] = []
    for elem in elements.values():
        if not isinstance(elem, dict):
            continue
        etype = elem.get("type", "")
        # Skip child elements (attributes, methods, bodies)
        if etype in _CHILD_ELEMENT_TYPES:
            continue
        # Keep primary elements or anything with an owner == null
        owner = elem.get("owner")
        is_primary = etype in primary_types or (not isinstance(owner, str) or not owner)
        if not is_primary:
            continue

        bounds = elem.get("bounds")
        if isinstance(bounds, dict):
            try:
                x = int(round(float(bounds["x"])))
                y = int(round(float(bounds["y"])))
                w = int(round(float(bounds.get("width", CLASS_WIDTH))))
                h = int(round(float(bounds.get("height", CLASS_MIN_HEIGHT))))
                rects.append(Rect(x, y, w, h))
            except (KeyError, TypeError, ValueError):
                continue
    return rects


# ---------------------------------------------------------------------------
# Core placement algorithm
# ---------------------------------------------------------------------------

def _find_free_position(
    width: int,
    height: int,
    occupied: List[Rect],
    preferred_x: int = CANVAS_MIN_X,
    preferred_y: int = CANVAS_MIN_Y,
    scan_direction: str = "right-then-down",
    canvas_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int]:
    """Find the first non-overlapping position using a scanning strategy.

    Starts near (preferred_x, preferred_y) and scans outward.
    canvas_bounds: optional (min_x, max_x, min_y, max_y) for dynamic sizing.
    """
    c_min_x, c_max_x, c_min_y, c_max_y = canvas_bounds or (
        CANVAS_MIN_X, CANVAS_MAX_X, CANVAS_MIN_Y, CANVAS_MAX_Y,
    )
    step_x = width + H_GAP
    step_y = height + V_GAP

    # Try the preferred position first
    candidate = Rect(_snap(preferred_x), _snap(preferred_y), width, height)
    if not _collides(candidate, occupied):
        return candidate.x, candidate.y

    # Spiral outward from the preferred position
    for ring in range(1, 60):
        for dx_mult in range(-ring, ring + 1):
            for dy_mult in range(-ring, ring + 1):
                if abs(dx_mult) != ring and abs(dy_mult) != ring:
                    continue  # only check the ring perimeter
                cx = _snap(preferred_x + dx_mult * step_x)
                cy = _snap(preferred_y + dy_mult * step_y)
                # Keep within canvas bounds
                if cx < c_min_x or cx + width > c_max_x:
                    continue
                if cy < c_min_y or cy + height > c_max_y:
                    continue
                candidate = Rect(cx, cy, width, height)
                if not _collides(candidate, occupied):
                    return cx, cy

    # Last-resort fallback: place at preferred position anyway
    return _snap(preferred_x), _snap(preferred_y)


def _collides(rect: Rect, occupied: List[Rect]) -> bool:
    """Check whether *rect* overlaps any occupied rectangle (with margin)."""
    expanded = rect.expanded(MARGIN)
    return any(expanded.overlaps(occ) for occ in occupied)


# ---------------------------------------------------------------------------
# Grid helpers for relationship-aware layout
# ---------------------------------------------------------------------------

def _nearest_free_grid_cell(
    grid: Dict[Tuple[int, int], str],
    preferred_row: int,
    preferred_col: int,
) -> Tuple[int, int]:
    """Return the free grid cell closest to *(preferred_row, preferred_col)*.

    Ties are broken by preferring non-negative coordinates (grow right and
    down rather than left and up) and then by (row, col) order.
    """
    if (preferred_row, preferred_col) not in grid:
        return (preferred_row, preferred_col)
    for ring in range(1, 30):
        best: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple] = None
        for dr in range(-ring, ring + 1):
            for dc in range(-ring, ring + 1):
                if abs(dr) != ring and abs(dc) != ring:
                    continue
                r = preferred_row + dr
                c = preferred_col + dc
                if (r, c) in grid:
                    continue
                dist = abs(dr) + abs(dc)
                neg_row = 0 if r >= 0 else 1
                neg_col = 0 if c >= 0 else 1
                key = (dist, neg_row, neg_col, r, c)
                if best_key is None or key < best_key:
                    best = (r, c)
                    best_key = key
        if best is not None:
            return best
    return (preferred_row, preferred_col)


def _nearest_free_cell_below(
    grid: Dict[Tuple[int, int], str],
    parent_row: int,
    parent_col: int,
) -> Tuple[int, int]:
    """Find the nearest free grid cell *below* *(parent_row, parent_col)*.

    Scans rows below the parent, preferring the same column and limiting
    horizontal spread to ±2 columns per row.  For wide inheritance trees
    (5+ children), this creates a compact 2-row block instead of a long
    single row.
    """
    max_col_spread = 2  # allow up to ±2 columns before moving to next row
    for row in range(parent_row + 1, parent_row + 10):
        if (row, parent_col) not in grid:
            return (row, parent_col)
        for dc in range(1, max_col_spread + 1):
            if (row, parent_col + dc) not in grid:
                return (row, parent_col + dc)
            if (row, parent_col - dc) not in grid:
                return (row, parent_col - dc)
    return (parent_row + 1, parent_col)


def _best_neighbor_grid_cell(
    grid: Dict[Tuple[int, int], str],
    placed_neighbors: List[Tuple[str, Tuple[int, int]]],
) -> Tuple[int, int]:
    """Pick the free grid cell adjacent to *placed_neighbors* that keeps
    the layout compact.

    Candidates are the four cardinal neighbours of every already-placed
    neighbour.  They are ranked by:

    1. Manhattan distance to the *neighbour centroid* (keeps related
       elements close).
    2. Manhattan distance to the *overall grid centroid* (keeps the whole
       diagram compact instead of growing in one direction).
    3. Prefer non-negative row / col (grow right-then-down).
    4. Deterministic (row, col) tiebreaker.
    """
    n_avg_row = sum(pos[0] for _, pos in placed_neighbors) / len(placed_neighbors)
    n_avg_col = sum(pos[1] for _, pos in placed_neighbors) / len(placed_neighbors)

    all_positions = list(grid.keys())
    g_avg_row = sum(r for r, _ in all_positions) / len(all_positions)
    g_avg_col = sum(c for _, c in all_positions) / len(all_positions)

    candidates: Set[Tuple[int, int]] = set()
    for _, (nr, nc) in placed_neighbors:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            cell = (nr + dr, nc + dc)
            if cell not in grid:
                candidates.add(cell)

    if not candidates:
        # Widen search to radius-2 neighbours before falling back
        for _, (nr, nc) in placed_neighbors:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    cell = (nr + dr, nc + dc)
                    if cell not in grid:
                        candidates.add(cell)

    if not candidates:
        # Last resort: find nearest free cell to the neighbour centroid
        return _nearest_free_grid_cell(grid, round(n_avg_row), round(n_avg_col))

    def _score(cell: Tuple[int, int]):
        r, c = cell
        neighbour_dist = abs(r - n_avg_row) + abs(c - n_avg_col)
        global_dist = abs(r - g_avg_row) + abs(c - g_avg_col)
        neg_row = 0 if r >= 0 else 1
        neg_col = 0 if c >= 0 else 1
        return (neighbour_dist, global_dist, neg_row, neg_col, r, c)

    return min(candidates, key=_score)


def _compact_grid(
    grid: Dict[Tuple[int, int], str],
) -> Tuple[Dict[Tuple[int, int], str], Dict[str, Tuple[int, int]]]:
    """Remove empty rows and columns from the grid.

    Returns a new grid and name_to_grid with consecutive row/col indices.
    This eliminates gaps that cause unnecessary spread in the pixel layout.
    """
    if not grid:
        return grid, {}

    # Collect used rows and columns in sorted order
    used_rows = sorted({r for r, _ in grid})
    used_cols = sorted({c for _, c in grid})

    # Build remapping: old index → new consecutive index
    row_map = {old: new for new, old in enumerate(used_rows)}
    col_map = {old: new for new, old in enumerate(used_cols)}

    new_grid: Dict[Tuple[int, int], str] = {}
    new_name_to_grid: Dict[str, Tuple[int, int]] = {}

    for (r, c), name in grid.items():
        new_cell = (row_map[r], col_map[c])
        new_grid[new_cell] = name
        new_name_to_grid[name] = new_cell

    return new_grid, new_name_to_grid


# ---------------------------------------------------------------------------
# Edge direction computation (shared by all diagram types)
# ---------------------------------------------------------------------------

def _compute_edge_directions(
    edges: List[Dict[str, Any]],
    element_info: Dict[str, Tuple[Dict[str, Any], Tuple[int, int]]],
) -> None:
    """Compute ``sourceDirection`` / ``targetDirection`` for every edge.

    Parameters
    ----------
    edges : list[dict]
        Relationship or transition dicts.  Each must have ``source`` and
        ``target`` keys that are element-name strings.  The function
        **mutates** each dict by adding ``sourceDirection`` and
        ``targetDirection`` (one of ``"Left"``, ``"Right"``, ``"Up"``,
        ``"Down"``).
    element_info : dict[str, (position_dict, (width, height))]
        Mapping from element name → (its ``position`` dict, its pixel
        ``(width, height)``).  Position dicts must have ``x`` and ``y``.
    """
    for edge in edges:
        src_name = edge.get("source", "")
        tgt_name = edge.get("target", "")
        src_info = element_info.get(src_name)
        tgt_info = element_info.get(tgt_name)
        if not src_info or not tgt_info:
            continue

        src_pos, (src_w, src_h) = src_info
        tgt_pos, (tgt_w, tgt_h) = tgt_info

        # Centre of each element
        src_cx = src_pos.get("x", 0) + src_w / 2
        src_cy = src_pos.get("y", 0) + src_h / 2
        tgt_cx = tgt_pos.get("x", 0) + tgt_w / 2
        tgt_cy = tgt_pos.get("y", 0) + tgt_h / 2

        dx = tgt_cx - src_cx
        dy = tgt_cy - src_cy

        etype = (edge.get("type") or "").lower()
        if etype in ("inheritance", "generalization"):
            # Inheritance: child always points Up toward parent
            if dy < 0:
                edge["sourceDirection"] = "Up"
                edge["targetDirection"] = "Down"
            else:
                edge["sourceDirection"] = "Down"
                edge["targetDirection"] = "Up"
        else:
            # General rule: pick the axis with the larger delta
            if abs(dx) >= abs(dy):
                if dx >= 0:
                    edge["sourceDirection"] = "Right"
                    edge["targetDirection"] = "Left"
                else:
                    edge["sourceDirection"] = "Left"
                    edge["targetDirection"] = "Right"
            else:
                if dy >= 0:
                    edge["sourceDirection"] = "Down"
                    edge["targetDirection"] = "Up"
                else:
                    edge["sourceDirection"] = "Up"
                    edge["targetDirection"] = "Down"


# ---------------------------------------------------------------------------
# Shared grid → pixel helpers (used by class, object, state layouts)
# ---------------------------------------------------------------------------

def _build_edge_pairs(
    edges: List[Dict[str, Any]],
    element_lookup: Dict[str, Any],
) -> Set[Tuple[str, str]]:
    """Build a set of canonical (name1, name2) pairs from edge dicts.

    Works for relationships, links, and transitions — any dict with
    ``source`` and ``target`` keys.
    """
    pairs: Set[Tuple[str, str]] = set()
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src in element_lookup and tgt in element_lookup:
            pair = (min(src, tgt), max(src, tgt))
            pairs.add(pair)
    return pairs


def _grid_to_pixel_positions(
    grid: Dict[Tuple[int, int], str],
    sizes: Dict[str, Tuple[int, int]],
    element_lookup: Dict[str, Dict[str, Any]],
    occupied: List[Rect],
    canvas_bounds: Tuple[int, int, int, int],
    default_size: Tuple[int, int],
    edge_pairs: Optional[Set[Tuple[str, str]]] = None,
    n_elements: int = 0,
) -> None:
    """Convert logical grid positions to pixel coordinates.

    Mutates each element dict in *element_lookup* by setting its
    ``position`` key.  Also appends placed :class:`Rect` instances to
    *occupied* so subsequent single-element placements avoid collisions.

    Parameters
    ----------
    grid : dict[(row, col) → element_name]
    sizes : dict[name → (width, height)]
    element_lookup : dict[name → element dict]  (mutated: ``position`` set)
    occupied : list[Rect]  (mutated: new rects appended)
    canvas_bounds : (min_x, max_x, min_y, max_y)
    default_size : fallback (width, height)
    edge_pairs : optional set of connected name pairs (unused for now,
        reserved for future edge-aware gap adjustment)
    n_elements : total element count (for future scaling)
    """
    if not grid:
        return

    # --- Grid bounds ---
    min_row = min(r for r, _ in grid)
    max_row = max(r for r, _ in grid)
    min_col = min(c for _, c in grid)
    max_col = max(c for _, c in grid)

    n_rows = max_row - min_row + 1
    n_cols = max_col - min_col + 1

    # --- Per-column widths and per-row heights ---
    col_widths: Dict[int, int] = {}
    row_heights: Dict[int, int] = {}

    for (r, c), name in grid.items():
        w, h = sizes.get(name, default_size)
        col_widths[c] = max(col_widths.get(c, 0), w)
        row_heights[r] = max(row_heights.get(r, 0), h)

    # --- Compact gap calculation ---
    h_gap = 60
    v_gap = 50

    # --- Total layout dimensions ---
    total_width = sum(col_widths.get(c, default_size[0]) for c in range(min_col, max_col + 1))
    total_width += h_gap * max(0, n_cols - 1)

    total_height = sum(row_heights.get(r, default_size[1]) for r in range(min_row, max_row + 1))
    total_height += v_gap * max(0, n_rows - 1)

    # --- Center on origin ---
    origin_x = _snap(-total_width // 2)
    origin_y = _snap(-total_height // 2)

    # --- Assign pixel coordinates ---
    for (r, c), name in grid.items():
        elem = element_lookup.get(name)
        if not elem:
            continue

        w, h = sizes.get(name, default_size)

        # X: sum of column widths + gaps for columns before this one
        px = origin_x
        for cc in range(min_col, c):
            px += col_widths.get(cc, default_size[0]) + h_gap

        # Center element within its column cell
        col_w = col_widths.get(c, default_size[0])
        px += (col_w - w) // 2

        # Y: sum of row heights + gaps for rows before this one
        py = origin_y
        for rr in range(min_row, r):
            py += row_heights.get(rr, default_size[1]) + v_gap

        # Center element within its row cell
        row_h = row_heights.get(r, default_size[1])
        py += (row_h - h) // 2

        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=_snap(px),
                                    preferred_y=_snap(py),
                                    canvas_bounds=canvas_bounds)
        elem["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))


# ---------------------------------------------------------------------------
# Public layout functions per diagram type
# ---------------------------------------------------------------------------

def layout_class_single(
    spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign a ``position`` to a single class element spec.

    Returns the spec (mutated in place) with ``position: {x, y}``.
    """
    width, height = estimate_class_size(spec)
    occupied = extract_occupied_rects(existing_model, "ClassDiagram")
    center_x = _snap((CANVAS_MIN_X + CANVAS_MAX_X) // 2 - width // 2)
    center_y = _snap((CANVAS_MIN_Y + CANVAS_MAX_Y) // 2 - height // 2)
    x, y = _find_free_position(width, height, occupied,
                                preferred_x=center_x, preferred_y=center_y)
    spec["position"] = {"x": x, "y": y}
    return spec


# ---------------------------------------------------------------------------
# Sugiyama hierarchical layout helpers (used by layout_class_system)
# ---------------------------------------------------------------------------


def _sugiyama_build_graph(
    class_names: Dict[str, Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, Set[str]],   # successors
    Dict[str, Set[str]],   # predecessors
    Dict[str, str],        # parent_of  (child -> parent for inheritance)
    Dict[str, str],        # owner_of   (part -> whole for composition)
    Dict[str, Set[str]],   # adjacency  (undirected)
    List[Tuple[str, str, int]],  # directed edges with priority
]:
    """Phase 0 – Build a directed graph from UML relationships.

    Returns successors, predecessors, parent_of, owner_of, undirected
    adjacency, and a list of directed edges (src, tgt, priority) where
    lower priority numbers mean the edge direction is more important
    to preserve (inheritance=0, composition=1, other=2).
    """
    successors: Dict[str, Set[str]] = defaultdict(set)
    predecessors: Dict[str, Set[str]] = defaultdict(set)
    parent_of: Dict[str, str] = {}
    owner_of: Dict[str, str] = {}
    adjacency: Dict[str, Set[str]] = {name: set() for name in class_names}
    directed_edges: List[Tuple[str, str, int]] = []

    # Compute degree for heuristic direction of association edges
    degree: Dict[str, int] = defaultdict(int)
    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        if src in class_names and tgt in class_names:
            degree[src] += 1
            degree[tgt] += 1

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rtype = (rel.get("type") or "").lower()
        if src not in class_names or tgt not in class_names:
            continue
        if src == tgt:
            continue

        adjacency[src].add(tgt)
        adjacency[tgt].add(src)

        if rtype in ("inheritance", "generalization"):
            # source = child, target = parent  => edge: parent -> child
            parent_of[src] = tgt
            successors[tgt].add(src)
            predecessors[src].add(tgt)
            directed_edges.append((tgt, src, 0))
        elif rtype in ("composition", "aggregation"):
            # source = whole, target = part  => edge: whole -> part
            owner_of[tgt] = src
            successors[src].add(tgt)
            predecessors[tgt].add(src)
            directed_edges.append((src, tgt, 1))
        else:
            # Association / other: higher-degree node above lower-degree node
            if degree[src] >= degree[tgt]:
                successors[src].add(tgt)
                predecessors[tgt].add(src)
                directed_edges.append((src, tgt, 2))
            else:
                successors[tgt].add(src)
                predecessors[src].add(tgt)
                directed_edges.append((tgt, src, 2))

    return successors, predecessors, parent_of, owner_of, adjacency, directed_edges


def _sugiyama_remove_cycles(
    nodes: List[str],
    successors: Dict[str, Set[str]],
    predecessors: Dict[str, Set[str]],
    directed_edges: List[Tuple[str, str, int]],
) -> None:
    """Phase 1 – Remove cycles via iterative DFS, reversing back-edges.

    High-priority edges (inheritance/composition) are preferred to keep
    in their original direction.  Mutates *successors* and *predecessors*
    in place.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in nodes}
    back_edges: List[Tuple[str, str]] = []

    # Sort starting nodes: prefer nodes with no predecessors (roots)
    start_order = sorted(nodes, key=lambda n: (len(predecessors.get(n, set())), n))

    for start in start_order:
        if color[start] != WHITE:
            continue
        # Iterative DFS using explicit stack of (node, iterator)
        stack: List[Tuple[str, List[str], int]] = []
        color[start] = GRAY
        neighbors = sorted(successors.get(start, set()))
        stack.append((start, neighbors, 0))

        while stack:
            node, nbrs, idx = stack[-1]
            if idx < len(nbrs):
                stack[-1] = (node, nbrs, idx + 1)
                nxt = nbrs[idx]
                if color.get(nxt, WHITE) == GRAY:
                    # Back edge found
                    back_edges.append((node, nxt))
                elif color.get(nxt, WHITE) == WHITE:
                    color[nxt] = GRAY
                    nxt_nbrs = sorted(successors.get(nxt, set()))
                    stack.append((nxt, nxt_nbrs, 0))
            else:
                color[node] = BLACK
                stack.pop()

    # Reverse back edges (remove cycle)
    for u, v in back_edges:
        if v in successors.get(u, set()):
            successors[u].discard(v)
            predecessors[v].discard(u)
            successors[v].add(u)
            predecessors[u].add(v)


def _sugiyama_assign_layers(
    nodes: List[str],
    successors: Dict[str, Set[str]],
    predecessors: Dict[str, Set[str]],
) -> Dict[str, int]:
    """Phase 2 – Assign layers using longest-path + Kahn's topological sort.

    Sources (no predecessors) get layer 0.  Each node's layer is
    max(predecessor layers) + 1.  Isolated nodes go to the middle layer.
    """
    # Kahn's algorithm for topological sort
    in_degree: Dict[str, int] = {n: 0 for n in nodes}
    for n in nodes:
        for s in successors.get(n, set()):
            if s in in_degree:
                in_degree[s] += 1

    queue: deque = deque()
    for n in sorted(nodes):  # sorted for determinism
        if in_degree[n] == 0:
            queue.append(n)

    topo_order: List[str] = []
    layer: Dict[str, int] = {}

    while queue:
        n = queue.popleft()
        topo_order.append(n)
        # Layer = max of predecessor layers + 1, or 0 if no predecessors
        pred_layers = [layer[p] for p in predecessors.get(n, set())
                       if p in layer]
        layer[n] = (max(pred_layers) + 1) if pred_layers else 0

        for s in sorted(successors.get(n, set())):
            if s in in_degree:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)

    # Handle nodes not reached by topo sort (remaining cycles, shouldn't happen
    # after cycle removal but just in case)
    for n in nodes:
        if n not in layer:
            layer[n] = 0

    # Compact layers: pull nodes up to min_allowed = max(pred layers) + 1
    for n in topo_order:
        if n not in layer:
            continue
        pred_layers = [layer[p] for p in predecessors.get(n, set())
                       if p in layer]
        min_allowed = (max(pred_layers) + 1) if pred_layers else 0
        layer[n] = min_allowed

    # Isolated nodes (no edges at all) go to the middle layer
    max_layer = max(layer.values()) if layer else 0
    mid_layer = max_layer // 2
    for n in nodes:
        has_edges = bool(successors.get(n, set())) or bool(predecessors.get(n, set()))
        if not has_edges:
            layer[n] = mid_layer

    # --- Layer balancing: split overly wide layers ---
    # If a layer has more than max_per_layer nodes AND some can be pushed
    # down without violating constraints, redistribute to reduce width.
    import math as _math
    max_per_layer = max(3, int(_math.ceil(_math.sqrt(len(nodes)))))

    # Build layers dict temporarily
    _layers_tmp: Dict[int, List[str]] = defaultdict(list)
    for n, li in layer.items():
        _layers_tmp[li].append(n)

    changed = True
    iterations = 0
    while changed and iterations < 10:
        changed = False
        iterations += 1
        for li in sorted(_layers_tmp.keys()):
            if len(_layers_tmp[li]) <= max_per_layer:
                continue
            # Try to push some nodes down to a new sub-layer
            # Candidates: nodes that have NO successors in layer li+1
            # that depend on them being in layer li
            movable = []
            for n in _layers_tmp[li]:
                # Can move n to li+1 if none of its successors are in li+1
                # AND all its predecessors are in layers < li+1
                succs_in_next = [s for s in successors.get(n, set())
                                 if layer.get(s) == li + 1]
                # Don't move nodes that have inheritance children expecting
                # them to stay in this layer
                if not succs_in_next:
                    movable.append(n)
            if not movable:
                continue
            # Move excess nodes (prefer those with fewer predecessors in this layer)
            n_to_move = len(_layers_tmp[li]) - max_per_layer
            movable.sort(key=lambda n: (
                len([p for p in predecessors.get(n, set()) if layer.get(p, -1) == li]),
                n
            ))
            moved = movable[:n_to_move]
            if not moved:
                continue
            # Shift all layers >= li+1 down by 1 to make room
            for n2 in list(layer.keys()):
                if layer[n2] > li:
                    layer[n2] += 1
            # Rebuild temp layers
            for m in moved:
                layer[m] = li + 1
            _layers_tmp = defaultdict(list)
            for n2, l2 in layer.items():
                _layers_tmp[l2].append(n2)
            changed = True
            break

    return layer


def _sugiyama_minimize_crossings(
    layers: Dict[int, List[str]],
    successors: Dict[str, Set[str]],
    predecessors: Dict[str, Set[str]],
    parent_of: Dict[str, str],
    num_sweeps: int = 4,
) -> Dict[int, List[str]]:
    """Phase 3 – Barycenter crossing minimization with sibling grouping.

    Performs *num_sweeps* alternating top-down / bottom-up sweeps.
    After sweeps, groups inheritance siblings contiguously and centers
    them under their parent.
    """
    if not layers:
        return layers

    layer_indices = sorted(layers.keys())

    # Build position lookup
    def _pos_map(layers_dict: Dict[int, List[str]]) -> Dict[str, int]:
        pm: Dict[str, int] = {}
        for _li, nodes_in_layer in layers_dict.items():
            for idx, nd in enumerate(nodes_in_layer):
                pm[nd] = idx
        return pm

    for sweep in range(num_sweeps):
        if sweep % 2 == 0:
            # Top-down sweep
            for li_idx in range(1, len(layer_indices)):
                li = layer_indices[li_idx]
                prev_li = layer_indices[li_idx - 1]
                pos = _pos_map(layers)
                barycenters: Dict[str, float] = {}
                for node in layers[li]:
                    # Neighbors in the previous (upper) layer
                    upper_neighbors = [
                        p for p in predecessors.get(node, set())
                        if p in pos and p in set(layers.get(prev_li, []))
                    ]
                    if upper_neighbors:
                        barycenters[node] = sum(pos[p] for p in upper_neighbors) / len(upper_neighbors)
                    else:
                        barycenters[node] = float(pos.get(node, 0))
                layers[li] = sorted(layers[li], key=lambda nd: barycenters.get(nd, 0.0))
        else:
            # Bottom-up sweep
            for li_idx in range(len(layer_indices) - 2, -1, -1):
                li = layer_indices[li_idx]
                next_li = layer_indices[li_idx + 1]
                pos = _pos_map(layers)
                barycenters: Dict[str, float] = {}
                for node in layers[li]:
                    lower_neighbors = [
                        s for s in successors.get(node, set())
                        if s in pos and s in set(layers.get(next_li, []))
                    ]
                    if lower_neighbors:
                        barycenters[node] = sum(pos[s] for s in lower_neighbors) / len(lower_neighbors)
                    else:
                        barycenters[node] = float(pos.get(node, 0))
                layers[li] = sorted(layers[li], key=lambda nd: barycenters.get(nd, 0.0))

    # Group inheritance siblings contiguously under their parent
    # Build parent -> children mapping per layer
    children_of: Dict[str, List[str]] = defaultdict(list)
    for child, parent in parent_of.items():
        children_of[parent].append(child)

    for li in layer_indices:
        current = layers[li]
        # Find siblings in this layer
        sibling_groups: Dict[str, List[str]] = defaultdict(list)
        for node in current:
            p = parent_of.get(node)
            if p and p in children_of:
                sibling_groups[p].append(node)

        if not sibling_groups:
            continue

        # For each parent, ensure its children appear contiguously
        pos = _pos_map(layers)
        for parent, children in sibling_groups.items():
            if len(children) < 2:
                continue
            # Find the parent's position in its layer to compute ideal center
            parent_pos = pos.get(parent, 0)
            # Remove children from current layer, reinsert contiguously
            others = [n for n in current if n not in children]
            # Determine insertion point: closest to parent_pos
            best_insert = 0
            if others:
                # Insert near the position that centers children under parent
                for i in range(len(others) + 1):
                    best_insert = i
                    # Heuristic: insert where avg position would be closest to parent_pos
                    if i < len(others) and pos.get(others[i], 0) > parent_pos:
                        break
            # Sort children deterministically
            children_sorted = sorted(children, key=lambda c: pos.get(c, 0))
            new_layer = others[:best_insert] + children_sorted + others[best_insert:]
            layers[li] = new_layer

    return layers


def _sugiyama_assign_coordinates(
    layers: Dict[int, List[str]],
    sizes: Dict[str, Tuple[int, int]],
    parent_of: Dict[str, str],
    owner_of: Dict[str, str],
    default_size: Tuple[int, int],
    successors: Optional[Dict[str, Set[str]]] = None,
    predecessors: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Tuple[int, int]]:
    """Phase 4 – Convert layer assignments to pixel coordinates.

    Y is computed from cumulative layer heights + v_gap (50px).
    X is sequential within each layer with h_gap (60px), then each
    layer is centered.  Inheritance children are shifted to center
    under their parent.  The entire layout is centered on origin (0, 0).
    All coordinates are snapped to the grid.
    """
    if successors is None:
        successors = {}
    if predecessors is None:
        predecessors = {}
    if not layers:
        return {}

    h_gap = 60
    v_gap = 50

    layer_indices = sorted(layers.keys())

    # Compute per-layer max height and cumulative Y offsets
    layer_heights: Dict[int, int] = {}
    for li in layer_indices:
        max_h = 0
        for node in layers[li]:
            _, h = sizes.get(node, default_size)
            max_h = max(max_h, h)
        layer_heights[li] = max_h

    # Y offset for each layer (cumulative)
    layer_y: Dict[int, int] = {}
    cum_y = 0
    for li in layer_indices:
        layer_y[li] = cum_y
        cum_y += layer_heights[li] + v_gap

    # Compute per-node X positions within each layer, then center the layer
    positions: Dict[str, Tuple[int, int]] = {}
    layer_widths: Dict[int, int] = {}

    # Adaptive h_gap: reduce gap for wide layers (>4 nodes)
    max_layer_size = max(len(layers[li]) for li in layer_indices) if layer_indices else 1

    for li in layer_indices:
        nodes = layers[li]
        if not nodes:
            continue
        # Use smaller gap for wider layers
        layer_h_gap = h_gap if len(nodes) <= 3 else max(30, h_gap - (len(nodes) - 3) * 8)
        # Sequential X placement
        x_positions: List[int] = []
        cur_x = 0
        for node in nodes:
            w, _ = sizes.get(node, default_size)
            x_positions.append(cur_x)
            cur_x += w + layer_h_gap
        # Total layer width
        last_node = nodes[-1]
        last_w, _ = sizes.get(last_node, default_size)
        total_layer_width = x_positions[-1] + last_w if x_positions else 0
        layer_widths[li] = total_layer_width

        # Store raw x positions (before centering)
        for idx, node in enumerate(nodes):
            _, h = sizes.get(node, default_size)
            # Center vertically within the layer row
            y_offset = (layer_heights[li] - h) // 2
            positions[node] = (x_positions[idx], layer_y[li] + y_offset)

    # Center each layer horizontally: shift so each layer is centered at x=0
    max_width = max(layer_widths.values()) if layer_widths else 0
    for li in layer_indices:
        lw = layer_widths.get(li, 0)
        offset = (max_width - lw) // 2
        for node in layers[li]:
            old_x, old_y = positions[node]
            positions[node] = (old_x + offset, old_y)

    # Post-adjustment: shift inheritance children to center under parent
    children_of: Dict[str, List[str]] = defaultdict(list)
    for child, parent in parent_of.items():
        if child in positions and parent in positions:
            children_of[parent].append(child)

    for parent, children in children_of.items():
        if not children:
            continue
        parent_x, _ = positions[parent]
        parent_w, _ = sizes.get(parent, default_size)
        parent_center = parent_x + parent_w // 2

        # Compute children span center
        child_xs = [positions[c][0] for c in children]
        child_ws = [sizes.get(c, default_size)[0] for c in children]
        children_min_x = min(child_xs)
        children_max_x = max(child_xs[i] + child_ws[i] for i in range(len(children)))
        children_center = (children_min_x + children_max_x) // 2

        shift = parent_center - children_center
        if abs(shift) > 5:  # Only shift if meaningful
            # Check that shifting won't cause overlap with other nodes in the same layer
            # Get the layer for the children
            child_layer = None
            for li, nodes in layers.items():
                if children[0] in nodes:
                    child_layer = li
                    break
            if child_layer is not None:
                layer_nodes = layers[child_layer]
                child_set = set(children)
                # Check boundaries: find non-child neighbors
                can_shift = True
                for c in children:
                    new_cx = positions[c][0] + shift
                    cw, _ = sizes.get(c, default_size)
                    for other in layer_nodes:
                        if other in child_set:
                            continue
                        other_x, _ = positions[other]
                        other_w, _ = sizes.get(other, default_size)
                        # Check overlap
                        if new_cx < other_x + other_w + h_gap and new_cx + cw + h_gap > other_x:
                            can_shift = False
                            break
                    if not can_shift:
                        break
                if can_shift:
                    for c in children:
                        old_x, old_y = positions[c]
                        positions[c] = (old_x + shift, old_y)

    # Similarly shift composition parts near their owner
    parts_of: Dict[str, List[str]] = defaultdict(list)
    for part, whole in owner_of.items():
        if part in positions and whole in positions:
            parts_of[whole].append(part)

    for whole, parts in parts_of.items():
        if not parts:
            continue
        whole_x, _ = positions[whole]
        whole_w, _ = sizes.get(whole, default_size)
        whole_center = whole_x + whole_w // 2

        part_xs = [positions[p][0] for p in parts]
        part_ws = [sizes.get(p, default_size)[0] for p in parts]
        parts_min_x = min(part_xs)
        parts_max_x = max(part_xs[i] + part_ws[i] for i in range(len(parts)))
        parts_center = (parts_min_x + parts_max_x) // 2

        shift = whole_center - parts_center
        if abs(shift) > 5:
            # Similar overlap check
            part_layer = None
            for li, nodes in layers.items():
                if parts[0] in nodes:
                    part_layer = li
                    break
            if part_layer is not None:
                layer_nodes = layers[part_layer]
                part_set = set(parts)
                can_shift = True
                for p in parts:
                    new_px = positions[p][0] + shift
                    pw, _ = sizes.get(p, default_size)
                    for other in layer_nodes:
                        if other in part_set:
                            continue
                        other_x, _ = positions[other]
                        other_w, _ = sizes.get(other, default_size)
                        if new_px < other_x + other_w + h_gap and new_px + pw + h_gap > other_x:
                            can_shift = False
                            break
                    if not can_shift:
                        break
                if can_shift:
                    for p in parts:
                        old_x, old_y = positions[p]
                        positions[p] = (old_x + shift, old_y)

    # --- Post-processing: swap nodes within layers to reduce long edges ---
    # Build adjacency from successors/predecessors for edge-distance check
    import math as _math
    all_edges: List[Tuple[str, str]] = []
    all_adj: Dict[str, Set[str]] = defaultdict(set)
    for n in positions:
        for s in successors.get(n, set()):
            if s in positions:
                all_edges.append((n, s))
                all_adj[n].add(s)
                all_adj[s].add(n)
        for p in predecessors.get(n, set()):
            if p in positions:
                all_adj[n].add(p)

    def _edge_dist(a: str, b: str) -> float:
        ax, ay = positions[a]
        aw, ah = sizes.get(a, default_size)
        bx, by = positions[b]
        bw, bh = sizes.get(b, default_size)
        return _math.sqrt((ax + aw/2 - bx - bw/2)**2 + (ay + ah/2 - by - bh/2)**2)

    def _total_edge_cost() -> float:
        return sum(_edge_dist(a, b) for a, b in all_edges)

    # Try swapping pairs within the same layer to reduce total edge cost
    for _iteration in range(3):
        improved = False
        for li in layer_indices:
            nodes = layers[li]
            if len(nodes) < 2:
                continue
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    a, b = nodes[i], nodes[j]
                    # Only consider swapping if at least one has a long edge
                    a_max = max((_edge_dist(a, nb) for nb in all_adj.get(a, set()) if nb in positions), default=0)
                    b_max = max((_edge_dist(b, nb) for nb in all_adj.get(b, set()) if nb in positions), default=0)
                    if a_max < 350 and b_max < 350:
                        continue
                    # Swap positions
                    old_a, old_b = positions[a], positions[b]
                    # Swap X coordinates but keep Y (same layer)
                    positions[a] = (old_b[0], old_a[1])
                    positions[b] = (old_a[0], old_b[1])
                    new_a_max = max((_edge_dist(a, nb) for nb in all_adj.get(a, set()) if nb in positions), default=0)
                    new_b_max = max((_edge_dist(b, nb) for nb in all_adj.get(b, set()) if nb in positions), default=0)
                    if max(new_a_max, new_b_max) < max(a_max, b_max):
                        improved = True
                        # Keep the swap - update layer order too
                        nodes[i], nodes[j] = nodes[j], nodes[i]
                    else:
                        # Revert
                        positions[a] = old_a
                        positions[b] = old_b
        if not improved:
            break

    # --- Final overlap resolution: ensure no two nodes in the same layer
    #     overlap after all centering and swap post-processing. ---
    for li in layer_indices:
        layer_nodes = [n for n in layers[li] if n in positions]
        if len(layer_nodes) < 2:
            continue
        layer_nodes.sort(key=lambda n: positions[n][0])
        for idx in range(1, len(layer_nodes)):
            prev_name = layer_nodes[idx - 1]
            curr_name = layer_nodes[idx]
            prev_x, prev_y = positions[prev_name]
            curr_x, curr_y = positions[curr_name]
            prev_w = sizes.get(prev_name, default_size)[0]
            needed_x = prev_x + prev_w + h_gap
            if curr_x < needed_x:
                positions[curr_name] = (needed_x, curr_y)

    # Center entire layout on origin (0, 0)
    if positions:
        all_xs = [x for x, _ in positions.values()]
        all_ys = [y for _, y in positions.values()]
        all_ws = [sizes.get(n, default_size)[0] for n in positions]
        all_hs = [sizes.get(n, default_size)[1] for n in positions]

        min_x = min(all_xs)
        max_x = max(all_xs[i] + all_ws[i] for i in range(len(all_xs)))
        min_y = min(all_ys)
        max_y = max(all_ys[i] + all_hs[i] for i in range(len(all_ys)))

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        for node in list(positions.keys()):
            old_x, old_y = positions[node]
            positions[node] = (_snap(old_x - center_x), _snap(old_y - center_y))

    return positions


def layout_class_system(
    system_spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign positions to all classes in a complete class-diagram system spec.

    Uses a **Sugiyama-based hierarchical layout** algorithm with four phases:

    0. **Graph construction** – directed edges from UML relationships
       (inheritance parent→child, composition whole→part, association by
       degree heuristic).
    1. **Cycle removal** – iterative DFS reverses back-edges while
       preferring high-priority edge directions.
    2. **Layer assignment** – longest-path via Kahn's topological sort;
       isolated nodes placed in the middle layer.
    3. **Crossing minimization** – barycenter heuristic with alternating
       sweeps; inheritance siblings grouped contiguously.
    4. **Coordinate assignment** – per-layer heights/widths → pixel
       positions, children centered under parents, layout centered on
       origin, snapped to grid, overlaps resolved.

    Inheritance parents are guaranteed to appear above their children
    (smaller Y).  Composition owners appear above/beside their parts.
    """
    classes: List[Dict[str, Any]] = system_spec.get("classes", [])
    relationships: List[Dict[str, Any]] = system_spec.get("relationships", [])
    if not classes:
        return system_spec

    n_classes = len(classes)
    canvas_bounds = _dynamic_canvas_bounds(n_classes)
    occupied = extract_occupied_rects(existing_model, "ClassDiagram")

    # --- Build lookup ---
    class_names: Dict[str, Dict[str, Any]] = {
        c.get("className", ""): c for c in classes
    }
    all_nodes = list(class_names.keys())

    # --- Compute element sizes ---
    sizes: Dict[str, Tuple[int, int]] = {}
    for c in classes:
        name = c.get("className", "")
        sizes[name] = estimate_class_size(c)

    # ================================================================
    # Phase 0: Graph Construction
    # ================================================================
    (successors, predecessors, parent_of, owner_of,
     adjacency, directed_edges) = _sugiyama_build_graph(class_names, relationships)

    # ================================================================
    # Phase 1: Cycle Removal
    # ================================================================
    _sugiyama_remove_cycles(all_nodes, successors, predecessors, directed_edges)

    # ================================================================
    # Phase 2: Layer Assignment
    # ================================================================
    node_layer = _sugiyama_assign_layers(all_nodes, successors, predecessors)

    # Build layers dict: layer_index -> [nodes]
    layers: Dict[int, List[str]] = defaultdict(list)
    for node, li in node_layer.items():
        layers[li].append(node)
    # Sort each layer for determinism
    for li in layers:
        layers[li] = sorted(layers[li])

    # ================================================================
    # Phase 3: Crossing Minimization
    # ================================================================
    layers = _sugiyama_minimize_crossings(
        layers, successors, predecessors, parent_of, num_sweeps=8,
    )

    # ================================================================
    # Phase 4: Coordinate Assignment
    # ================================================================
    positions = _sugiyama_assign_coordinates(
        layers, sizes, parent_of, owner_of,
        default_size=(CLASS_WIDTH, CLASS_MIN_HEIGHT),
        successors=successors,
        predecessors=predecessors,
    )

    # --- Place elements, resolving overlaps with existing canvas elements ---
    for name, (px, py) in positions.items():
        spec = class_names.get(name)
        if not spec:
            continue
        w, h = sizes.get(name, (CLASS_WIDTH, CLASS_MIN_HEIGHT))
        x, y = _find_free_position(
            w, h, occupied,
            preferred_x=px, preferred_y=py,
            canvas_bounds=canvas_bounds,
        )
        spec["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))

    # --- Compute relationship connection directions ---
    _compute_edge_directions(
        relationships,
        {name: (spec.get("position", {}), sizes.get(name, (CLASS_WIDTH, CLASS_MIN_HEIGHT)))
         for name, spec in class_names.items()},
    )

    return system_spec


def layout_object_single(
    spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign position to a single object element."""
    width, height = estimate_object_size(spec)
    occupied = extract_occupied_rects(existing_model, "ObjectDiagram")
    center_x = _snap((CANVAS_MIN_X + CANVAS_MAX_X) // 2 - width // 2)
    center_y = _snap((CANVAS_MIN_Y + CANVAS_MAX_Y) // 2 - height // 2)
    x, y = _find_free_position(width, height, occupied,
                                preferred_x=center_x, preferred_y=center_y)
    spec["position"] = {"x": x, "y": y}
    return spec


def layout_object_system(
    system_spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign positions to all objects in a complete object diagram.

    Uses a **Sugiyama-based hierarchical layout** algorithm with four phases:

    0. **Graph construction** – directed edges from object links
       (higher-degree node above lower-degree node, like associations
       in class diagrams).
    1. **Cycle removal** – iterative DFS reverses back-edges.
    2. **Layer assignment** – longest-path via Kahn's topological sort;
       isolated nodes placed in the middle layer.
    3. **Crossing minimization** – barycenter heuristic with alternating
       sweeps.
    4. **Coordinate assignment** – per-layer heights/widths to pixel
       positions, layout centered on origin, snapped to grid, overlaps
       resolved.
    """
    objects: List[Dict[str, Any]] = system_spec.get("objects", [])
    if not objects:
        return system_spec

    links: List[Dict[str, Any]] = system_spec.get("links", [])
    n_objects = len(objects)
    canvas_bounds = _dynamic_canvas_bounds(n_objects)
    occupied = extract_occupied_rects(existing_model, "ObjectDiagram")

    # --- Build lookup ---
    obj_names: Dict[str, Dict[str, Any]] = {
        o.get("objectName", ""): o for o in objects
    }
    all_nodes = list(obj_names.keys())

    # --- Compute element sizes ---
    sizes: Dict[str, Tuple[int, int]] = {}
    for obj in objects:
        name = obj.get("objectName", "")
        sizes[name] = estimate_object_size(obj)

    # ================================================================
    # Phase 0: Graph Construction (manual — all links are associations)
    # ================================================================
    successors: Dict[str, Set[str]] = defaultdict(set)
    predecessors: Dict[str, Set[str]] = defaultdict(set)
    adjacency: Dict[str, Set[str]] = {name: set() for name in obj_names}
    directed_edges: List[Tuple[str, str, int]] = []

    # Compute degree for heuristic direction (higher-degree node on top)
    degree: Dict[str, int] = defaultdict(int)
    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        if src in obj_names and tgt in obj_names:
            degree[src] += 1
            degree[tgt] += 1

    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        if src not in obj_names or tgt not in obj_names:
            continue
        if src == tgt:
            continue

        adjacency[src].add(tgt)
        adjacency[tgt].add(src)

        # Higher-degree node above lower-degree node (priority 2 = association)
        if degree[src] >= degree[tgt]:
            successors[src].add(tgt)
            predecessors[tgt].add(src)
            directed_edges.append((src, tgt, 2))
        else:
            successors[tgt].add(src)
            predecessors[src].add(tgt)
            directed_edges.append((tgt, src, 2))

    # ================================================================
    # Phase 1: Cycle Removal
    # ================================================================
    _sugiyama_remove_cycles(all_nodes, successors, predecessors, directed_edges)

    # ================================================================
    # Phase 2: Layer Assignment
    # ================================================================
    node_layer = _sugiyama_assign_layers(all_nodes, successors, predecessors)

    # Build layers dict: layer_index -> [nodes]
    layers: Dict[int, List[str]] = defaultdict(list)
    for node, li in node_layer.items():
        layers[li].append(node)
    # Sort each layer for determinism
    for li in layers:
        layers[li] = sorted(layers[li])

    # ================================================================
    # Phase 3: Crossing Minimization
    # ================================================================
    # Object diagrams have no inheritance, so parent_of is empty
    parent_of: Dict[str, str] = {}
    layers = _sugiyama_minimize_crossings(
        layers, successors, predecessors, parent_of, num_sweeps=8,
    )

    # ================================================================
    # Phase 4: Coordinate Assignment
    # ================================================================
    # Object diagrams have no inheritance or composition, so both
    # parent_of and owner_of are empty dicts.
    owner_of: Dict[str, str] = {}
    positions = _sugiyama_assign_coordinates(
        layers, sizes, parent_of, owner_of,
        default_size=(OBJECT_WIDTH, OBJECT_MIN_HEIGHT),
        successors=successors,
        predecessors=predecessors,
    )

    # --- Place elements, resolving overlaps with existing canvas elements ---
    for name, (px, py) in positions.items():
        spec = obj_names.get(name)
        if not spec:
            continue
        w, h = sizes.get(name, (OBJECT_WIDTH, OBJECT_MIN_HEIGHT))
        x, y = _find_free_position(
            w, h, occupied,
            preferred_x=px, preferred_y=py,
            canvas_bounds=canvas_bounds,
        )
        spec["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))

    # --- Compute link directions ---
    _compute_edge_directions(
        links,
        {name: (spec.get("position", {}), sizes.get(name, (OBJECT_WIDTH, OBJECT_MIN_HEIGHT)))
         for name, spec in obj_names.items()},
    )

    return system_spec


def layout_state_single(
    spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign position to a single state element."""
    width, height = estimate_state_size(spec)
    occupied = extract_occupied_rects(existing_model, "StateMachineDiagram")
    center_x = _snap((CANVAS_MIN_X + CANVAS_MAX_X) // 2 - width // 2)
    center_y = _snap((CANVAS_MIN_Y + CANVAS_MAX_Y) // 2 - height // 2)
    x, y = _find_free_position(width, height, occupied,
                                preferred_x=center_x, preferred_y=center_y)
    spec["position"] = {"x": x, "y": y}
    return spec


def layout_state_system(
    system_spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign positions to a complete state machine using Sugiyama hierarchical layout.

    Uses the same four-phase Sugiyama algorithm as ``layout_class_system``,
    adapted for state-machine semantics:

    0. **Graph construction** -- directed edges follow transition direction
       (source -> target).  Built manually from transitions (no UML
       relationship heuristics needed).
    1. **Cycle removal** -- iterative DFS reverses back-edges.
    2. **Layer assignment** -- longest-path via Kahn's topological sort.
       Initial states are forced to layer 0; final states are forced to
       the maximum layer so the diagram flows top-to-bottom.
    3. **Crossing minimization** -- barycenter heuristic with alternating
       sweeps.
    4. **Coordinate assignment** -- per-layer heights/widths to pixel
       positions, layout centered on origin, snapped to grid, overlaps
       resolved against existing canvas elements.
    """
    states: List[Dict[str, Any]] = system_spec.get("states", [])
    if not states:
        return system_spec

    transitions: List[Dict[str, Any]] = system_spec.get("transitions", [])
    n_states = len(states)
    canvas_bounds = _dynamic_canvas_bounds(n_states)

    occupied = extract_occupied_rects(existing_model, "StateMachineDiagram")

    # --- Build state lookup ---
    state_names: Dict[str, Dict[str, Any]] = {
        s.get("stateName", ""): s for s in states
    }
    all_nodes = list(state_names.keys())

    # --- Categorize states ---
    initial_states: List[str] = []
    final_states: List[str] = []
    for s in states:
        name = s.get("stateName", "")
        stype = s.get("stateType", "regular")
        if stype == "initial":
            initial_states.append(name)
        elif stype == "final":
            final_states.append(name)

    # --- Compute element sizes ---
    sizes: Dict[str, Tuple[int, int]] = {}
    for s in states:
        name = s.get("stateName", "")
        sizes[name] = estimate_state_size(s)

    # ================================================================
    # Phase 0: Graph Construction (from transitions)
    # ================================================================
    successors: Dict[str, Set[str]] = defaultdict(set)
    predecessors: Dict[str, Set[str]] = defaultdict(set)
    directed_edges: List[Tuple[str, str, int]] = []

    for t in transitions:
        src = t.get("source", "")
        tgt = t.get("target", "")
        if src in state_names and tgt in state_names and src != tgt:
            successors[src].add(tgt)
            predecessors[tgt].add(src)
            directed_edges.append((src, tgt, 0))

    # ================================================================
    # Phase 1: Cycle Removal
    # ================================================================
    _sugiyama_remove_cycles(all_nodes, successors, predecessors, directed_edges)

    # ================================================================
    # Phase 2: Layer Assignment
    # ================================================================
    node_layer = _sugiyama_assign_layers(all_nodes, successors, predecessors)

    # Force initial states to layer 0
    for name in initial_states:
        if name in node_layer:
            node_layer[name] = 0

    # Force final states to the maximum layer
    if final_states:
        # Ensure final states are at least one layer below everything else
        non_final_max = max(
            (layer for n, layer in node_layer.items() if n not in final_states),
            default=0,
        )
        final_layer = max(max(node_layer.values()) if node_layer else 0,
                          non_final_max + 1)
        for name in final_states:
            if name in node_layer:
                node_layer[name] = final_layer

    # Build layers dict: layer_index -> [nodes]
    layers: Dict[int, List[str]] = defaultdict(list)
    for node, li in node_layer.items():
        layers[li].append(node)
    # Sort each layer for determinism
    for li in layers:
        layers[li] = sorted(layers[li])

    # ================================================================
    # Phase 3: Crossing Minimization
    # ================================================================
    # parent_of is empty for state machines (no inheritance hierarchy)
    layers = _sugiyama_minimize_crossings(
        layers, successors, predecessors, parent_of={}, num_sweeps=8,
    )

    # ================================================================
    # Phase 4: Coordinate Assignment
    # ================================================================
    # parent_of and owner_of are empty for state machines
    positions = _sugiyama_assign_coordinates(
        layers, sizes, parent_of={}, owner_of={},
        default_size=(STATE_WIDTH, STATE_MIN_HEIGHT),
        successors=successors,
        predecessors=predecessors,
    )

    # --- Place elements, resolving overlaps with existing canvas elements ---
    for name, (px, py) in positions.items():
        spec = state_names.get(name)
        if not spec:
            continue
        w, h = sizes.get(name, (STATE_WIDTH, STATE_MIN_HEIGHT))
        x, y = _find_free_position(
            w, h, occupied,
            preferred_x=px, preferred_y=py,
            canvas_bounds=canvas_bounds,
        )
        spec["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))

    # --- Compute transition directions ---
    _compute_edge_directions(
        transitions,
        {name: (spec.get("position", {}), sizes.get(name, (STATE_WIDTH, STATE_MIN_HEIGHT)))
         for name, spec in state_names.items()},
    )

    return system_spec


def layout_agent_single(
    spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign position to a single agent diagram element."""
    width, height = estimate_agent_element_size(spec)
    occupied = extract_occupied_rects(existing_model, "AgentDiagram")
    elem_type = spec.get("type", "state")
    # Intents go to the upper half, states to the lower half
    if elem_type == "intent":
        pref_y = _snap(CANVAS_MIN_Y + 60)
    elif elem_type == "initial":
        pref_y = _snap(CANVAS_MIN_Y + 40)
    else:
        pref_y = _snap((CANVAS_MIN_Y + CANVAS_MAX_Y) // 2)
    pref_x = _snap((CANVAS_MIN_X + CANVAS_MAX_X) // 2 - width // 2)
    x, y = _find_free_position(width, height, occupied,
                                preferred_x=pref_x, preferred_y=pref_y)
    spec["position"] = {"x": x, "y": y}
    return spec


def layout_agent_system(
    system_spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign positions to a complete agent diagram.

    Uses a **two-band hybrid layout** designed specifically for agent
    diagrams (cyclic state machines with a separate intent concept):

    **Band 1 (top)** -- Initial node + all intents in a horizontal row.
    **Band 2 (bottom)** -- States laid out with Sugiyama on the
    state-to-state transition subgraph only.

    This avoids the problems of pure Sugiyama on cyclic agent graphs
    where intents get scattered among state layers.
    """
    states_list: List[Dict[str, Any]] = system_spec.get("states", [])
    intents_list: List[Dict[str, Any]] = system_spec.get("intents", [])
    initial_nodes: List[Dict[str, Any]] = system_spec.get("initialNodes", [])
    agent_transitions: List[Dict[str, Any]] = system_spec.get("transitions", [])
    has_initial = bool(system_spec.get("hasInitialNode", False))

    n_total = len(states_list) + len(intents_list) + len(initial_nodes)
    if has_initial:
        n_total += 1
    if n_total == 0:
        return system_spec

    canvas_bounds = _dynamic_canvas_bounds(n_total)
    occupied = extract_occupied_rects(existing_model, "AgentDiagram")

    # --- Build unified lookup: name -> spec dict for every element ---
    element_lookup: Dict[str, Dict[str, Any]] = {}
    for node in initial_nodes:
        name = node.get("name", node.get("stateName", ""))
        if name:
            element_lookup[name] = node
    for intent in intents_list:
        name = intent.get("intentName", intent.get("name", ""))
        if name:
            element_lookup[name] = intent
    for state in states_list:
        name = state.get("stateName", state.get("name", ""))
        if name:
            element_lookup[name] = state

    # If hasInitialNode is set and "initial" is referenced in transitions
    # but not in the element lookup, create a synthetic initial node.
    # Store it in system_spec so the position propagates to the response.
    if has_initial and "initial" not in element_lookup:
        if not isinstance(system_spec.get("initialNode"), dict):
            system_spec["initialNode"] = {}
        system_spec["initialNode"]["type"] = "initial"
        element_lookup["initial"] = system_spec["initialNode"]

    all_nodes = list(element_lookup.keys())
    if not all_nodes:
        return system_spec

    # --- Compute sizes for all elements ---
    sizes: Dict[str, Tuple[int, int]] = {}
    intent_name_set: Set[str] = set()
    for intent in intents_list:
        iname = intent.get("intentName", intent.get("name", ""))
        if iname:
            intent_name_set.add(iname)

    for name, spec in element_lookup.items():
        etype = spec.get("type", "")
        if etype == "initial" or name == "initial":
            sizes[name] = (INITIAL_NODE_SIZE, INITIAL_NODE_SIZE)
        elif name in intent_name_set:
            sizes[name] = estimate_agent_element_size({"type": "intent", **spec})
        else:
            sizes[name] = estimate_agent_element_size({"type": "state", **spec})

    # --- Separate elements into bands ---
    initial_name: Optional[str] = None
    if has_initial and "initial" in element_lookup:
        initial_name = "initial"

    intent_names_ordered: List[str] = []
    for intent in intents_list:
        iname = intent.get("intentName", intent.get("name", ""))
        if iname and iname in element_lookup:
            intent_names_ordered.append(iname)

    state_names: List[str] = []
    for state in states_list:
        sname = state.get("stateName", state.get("name", ""))
        if sname and sname in element_lookup:
            state_names.append(sname)

    # ================================================================
    # Band 1: Intent band — initial node + intents in a horizontal row
    # ================================================================
    intent_h_gap = 60
    band1_positions: Dict[str, Tuple[int, int]] = {}
    band1_max_h = 0

    # Place initial node at (0, 0) — it will be on its own row or
    # at the start of the first intent row if there are few intents.
    intent_v_gap = 50
    max_intents_per_row = 4
    cur_x = 0
    cur_y = 0
    row_max_h = 0
    col_in_row = 0

    if initial_name is not None:
        iw, ih = sizes[initial_name]
        band1_positions[initial_name] = (cur_x, cur_y)
        row_max_h = max(row_max_h, ih)
        cur_x += iw + intent_h_gap
        col_in_row += 1

    # Place intents in rows of max_intents_per_row
    for iname in intent_names_ordered:
        if col_in_row >= max_intents_per_row:
            # Wrap to next row
            cur_y += row_max_h + intent_v_gap
            cur_x = 0
            row_max_h = 0
            col_in_row = 0
        iw, ih = sizes[iname]
        band1_positions[iname] = (cur_x, cur_y)
        row_max_h = max(row_max_h, ih)
        cur_x += iw + intent_h_gap
        col_in_row += 1

    band1_max_h = cur_y + row_max_h

    # ================================================================
    # Band 2: State band — Sugiyama on state-to-state transitions only
    # ================================================================
    band2_v_gap = 80  # gap between intent band and state band

    # Find the initial state (target of the initial -> state transition)
    initial_state_name: Optional[str] = None
    for t in agent_transitions:
        src = t.get("source", "")
        tgt = t.get("target", "")
        if src == "initial" and tgt in element_lookup and tgt not in intent_name_set:
            initial_state_name = tgt
            break

    band2_positions: Dict[str, Tuple[int, int]] = {}
    if state_names:
        # Build state-only subgraph (skip transitions involving "initial",
        # skip self-loops, skip intents)
        state_set = set(state_names)
        state_successors: Dict[str, Set[str]] = defaultdict(set)
        state_predecessors: Dict[str, Set[str]] = defaultdict(set)
        state_edges: List[Tuple[str, str, int]] = []

        for t in agent_transitions:
            src = t.get("source", "")
            tgt = t.get("target", "")
            if src == tgt:
                continue  # skip self-loops
            if src == "initial" or tgt == "initial":
                continue  # skip initial transitions
            if src not in state_set or tgt not in state_set:
                continue  # skip edges involving intents
            state_successors[src].add(tgt)
            state_predecessors[tgt].add(src)
            state_edges.append((src, tgt, 1))

        # Run Sugiyama phases on the state subgraph
        _sugiyama_remove_cycles(
            state_names, state_successors, state_predecessors, state_edges,
        )

        state_layer = _sugiyama_assign_layers(
            state_names, state_successors, state_predecessors,
        )

        # If there is an initial state, force it to layer 0
        if initial_state_name and initial_state_name in state_layer:
            init_layer = state_layer[initial_state_name]
            if init_layer != 0:
                # Shift all layers so the initial state is at layer 0
                for sn in state_layer:
                    state_layer[sn] -= init_layer

        # Build layers dict
        state_layers: Dict[int, List[str]] = defaultdict(list)
        for sname, li in state_layer.items():
            state_layers[li].append(sname)
        for li in state_layers:
            state_layers[li] = sorted(state_layers[li])

        # Crossing minimization
        state_layers = _sugiyama_minimize_crossings(
            state_layers, state_successors, state_predecessors,
            parent_of={}, num_sweeps=8,
        )

        # Coordinate assignment
        default_state_size = (AGENT_STATE_WIDTH, AGENT_NODE_MIN_HEIGHT)
        band2_positions = _sugiyama_assign_coordinates(
            state_layers, sizes, parent_of={}, owner_of={},
            default_size=default_state_size,
            successors=state_successors,
            predecessors=state_predecessors,
        )

        # Resolve within-layer overlaps caused by swap optimization or
        # centering.  Push overlapping nodes apart horizontally.
        min_gap = 60
        for li in sorted(state_layers.keys()):
            layer_nodes = [n for n in state_layers[li] if n in band2_positions]
            if len(layer_nodes) < 2:
                continue
            # Sort by x position
            layer_nodes.sort(key=lambda n: band2_positions[n][0])
            for idx in range(1, len(layer_nodes)):
                prev = layer_nodes[idx - 1]
                curr = layer_nodes[idx]
                prev_x, prev_y = band2_positions[prev]
                curr_x, curr_y = band2_positions[curr]
                prev_w = sizes.get(prev, default_state_size)[0]
                needed_x = prev_x + prev_w + min_gap
                if curr_x < needed_x:
                    band2_positions[curr] = (needed_x, curr_y)

    # ================================================================
    # Combine bands: place band 2 below band 1 with absolute offsets
    # ================================================================
    all_positions: Dict[str, Tuple[int, int]] = {}

    # Band 1 is already positioned starting at y=0.
    all_positions.update(band1_positions)

    # Band 2: normalize to start at y = band1_max_h + band2_v_gap
    if band2_positions:
        b2_min_x = min(x for x, _ in band2_positions.values())
        b2_min_y = min(y for _, y in band2_positions.values())
        state_y_start = band1_max_h + band2_v_gap
        for sname in band2_positions:
            sx, sy = band2_positions[sname]
            # Shift so band 2 top-left starts at (0, state_y_start)
            all_positions[sname] = (sx - b2_min_x, sy - b2_min_y + state_y_start)

    # Center horizontally only — preserve vertical band separation.
    # Vertical centering would pull states up into the intent band.
    if all_positions:
        pos_names = list(all_positions.keys())
        ds = (AGENT_STATE_WIDTH, AGENT_NODE_MIN_HEIGHT)
        all_xs = [all_positions[n][0] for n in pos_names]
        all_ws = [sizes.get(n, ds)[0] for n in pos_names]

        min_x = min(all_xs)
        max_x = max(all_xs[i] + all_ws[i] for i in range(len(all_xs)))
        cx = (min_x + max_x) // 2

        for name in pos_names:
            ox, oy = all_positions[name]
            all_positions[name] = (_snap(ox - cx), _snap(oy))

    # --- Place elements ---
    # Use computed positions directly (they're already non-overlapping).
    # Only fall back to _find_free_position if there are pre-existing
    # canvas elements that might collide.
    default_size = (AGENT_STATE_WIDTH, AGENT_NODE_MIN_HEIGHT)
    has_preexisting = len(occupied) > 0
    for name, (px, py) in all_positions.items():
        spec = element_lookup.get(name)
        if not spec:
            continue
        w, h = sizes.get(name, default_size)
        if has_preexisting:
            x, y = _find_free_position(
                w, h, occupied,
                preferred_x=px, preferred_y=py,
                canvas_bounds=canvas_bounds,
            )
        else:
            x, y = px, py
        spec["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))

    # --- Compute transition directions ---
    _compute_edge_directions(
        agent_transitions,
        {name: (spec.get("position", {}), sizes.get(name, default_size))
         for name, spec in element_lookup.items()},
    )

    return system_spec


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def apply_layout(
    spec: Dict[str, Any],
    diagram_type: str,
    mode: str = "single",
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """High-level dispatcher: pick the right layout function.

    Parameters
    ----------
    spec : dict
        The LLM-generated element or system specification (will be mutated).
    diagram_type : str
        One of ``ClassDiagram``, ``ObjectDiagram``, ``StateMachineDiagram``,
        ``AgentDiagram``.
    mode : str
        ``"single"`` for one element or ``"system"`` for a complete diagram.
    existing_model : dict, optional
        The current model JSON from the editor (used for collision avoidance).

    Returns
    -------
    dict
        The same *spec* with ``position`` fields filled in.
    """
    if diagram_type == "ClassDiagram":
        if mode == "system":
            return layout_class_system(spec, existing_model)
        return layout_class_single(spec, existing_model)

    if diagram_type == "ObjectDiagram":
        if mode == "system":
            return layout_object_system(spec, existing_model)
        return layout_object_single(spec, existing_model)

    if diagram_type == "StateMachineDiagram":
        if mode == "system":
            return layout_state_system(spec, existing_model)
        return layout_state_single(spec, existing_model)

    if diagram_type == "AgentDiagram":
        if mode == "system":
            return layout_agent_system(spec, existing_model)
        return layout_agent_single(spec, existing_model)

    # Fallback: try single-class layout
    return layout_class_single(spec, existing_model)
