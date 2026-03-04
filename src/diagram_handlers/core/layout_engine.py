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


def estimate_agent_element_size(spec: Dict[str, Any]) -> Tuple[int, int]:
    """Return (width, height) for an agent diagram element spec."""
    elem_type = spec.get("type", "state")
    if elem_type == "initial":
        return INITIAL_NODE_SIZE, INITIAL_NODE_SIZE
    if elem_type == "intent":
        n_phrases = len(spec.get("trainingPhrases", []))
        height = AGENT_NODE_MIN_HEIGHT + n_phrases * AGENT_PHRASE_ROW
        return AGENT_INTENT_WIDTH, max(height, AGENT_NODE_MIN_HEIGHT)
    # state (default)
    n_replies = len(spec.get("replies", []))
    n_fallback = len(spec.get("fallbackBodies", []))
    height = AGENT_NODE_MIN_HEIGHT + (n_replies + n_fallback) * AGENT_REPLY_ROW
    return AGENT_STATE_WIDTH, max(height, AGENT_NODE_MIN_HEIGHT)


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

    Scans successive rows below the parent, preferring the same column,
    then adjacent columns.  This guarantees inheritance children are
    always placed below their parent in the diagram.
    """
    for row in range(parent_row + 1, parent_row + 20):
        if (row, parent_col) not in grid:
            return (row, parent_col)
        for dc in range(1, 15):
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
        max_col = max(c for _, c in grid.keys()) if grid else 0
        return _nearest_free_grid_cell(grid, 0, max_col + 1)

    def _score(cell: Tuple[int, int]):
        r, c = cell
        neighbour_dist = abs(r - n_avg_row) + abs(c - n_avg_col)
        global_dist = abs(r - g_avg_row) + abs(c - g_avg_col)
        neg_row = 0 if r >= 0 else 1
        neg_col = 0 if c >= 0 else 1
        return (neighbour_dist, global_dist, neg_row, neg_col, r, c)

    return min(candidates, key=_score)


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


def layout_class_system(
    system_spec: Dict[str, Any],
    existing_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assign positions to all classes in a complete class-diagram system spec.

    Uses a **relationship-aware grid layout**: classes are assigned to
    logical grid cells via BFS from the most-connected hub, so that
    associated / composed / aggregated classes always occupy adjacent cells.
    Inheritance hierarchies flow top-to-bottom (child below parent).

    The grid is then converted to pixel coordinates with per-column widths
    and per-row heights, producing a clean, compact diagram with short edges.

    For large diagrams (>6 classes), the canvas bounds expand dynamically
    and the layout is centered on the origin.
    """
    classes: List[Dict[str, Any]] = system_spec.get("classes", [])
    relationships: List[Dict[str, Any]] = system_spec.get("relationships", [])
    if not classes:
        return system_spec

    # Dynamic canvas bounds for large diagrams
    n_classes = len(classes)
    canvas_bounds = _dynamic_canvas_bounds(n_classes)

    occupied = extract_occupied_rects(existing_model, "ClassDiagram")

    # --- Build graph structure ---
    class_names: Dict[str, Dict[str, Any]] = {
        c.get("className", ""): c for c in classes
    }
    parent_of: Dict[str, str] = {}          # child -> parent (Inheritance)
    adjacency: Dict[str, Set[str]] = {name: set() for name in class_names}

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rtype = (rel.get("type") or "").lower()
        if src in adjacency and tgt in adjacency:
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)
        if rtype in ("inheritance", "generalization"):
            parent_of[src] = tgt

    # --- Compute element sizes ---
    sizes: Dict[str, Tuple[int, int]] = {}
    for c in classes:
        name = c.get("className", "")
        sizes[name] = estimate_class_size(c)

    # --- BFS placement order (most-connected node first) ---
    placement_order: List[str] = []
    remaining = set(class_names.keys())
    while remaining:
        root = max(remaining,
                   key=lambda n: (len(adjacency.get(n, set())), n))
        queue = [root]
        bfs_seen: Set[str] = {root}
        while queue:
            current = queue.pop(0)
            placement_order.append(current)
            remaining.discard(current)
            neighbors = sorted(
                [n for n in adjacency.get(current, set())
                 if n not in bfs_seen and n in remaining],
                key=lambda n: (
                    0 if parent_of.get(n) == current else 1,  # inheritance children first
                    -len(adjacency.get(n, set())),            # then by degree descending
                    n,                                        # then alphabetical
                ),
            )
            for n in neighbors:
                bfs_seen.add(n)
                queue.append(n)

    # --- Assign logical grid cells ---
    # For large diagrams, limit the number of columns to keep the grid readable
    ideal_rows, ideal_cols = _ideal_grid_shape(n_classes)
    grid: Dict[Tuple[int, int], str] = {}
    name_to_grid: Dict[str, Tuple[int, int]] = {}

    for name in placement_order:
        if not grid:
            cell = (0, 0)
        else:
            p_name = parent_of.get(name)
            placed_neighbors = [
                (n, name_to_grid[n])
                for n in adjacency.get(name, set())
                if n in name_to_grid
            ]
            if p_name and p_name in name_to_grid:
                # Inheritance: child directly below parent
                pr, pc = name_to_grid[p_name]
                cell = _nearest_free_cell_below(grid, pr, pc)
            elif placed_neighbors:
                cell = _best_neighbor_grid_cell(grid, placed_neighbors)
            else:
                # Isolated class — wrap to next row if current row is full
                used_cols_in_row_0 = sum(1 for (r, _) in grid if r == 0)
                if used_cols_in_row_0 >= ideal_cols:
                    # Find the first row with space
                    for try_row in range(ideal_rows):
                        used_in_row = sum(1 for (r, _) in grid if r == try_row)
                        if used_in_row < ideal_cols:
                            max_c = max((c for r, c in grid if r == try_row), default=-1)
                            cell = _nearest_free_grid_cell(grid, try_row, max_c + 1)
                            break
                    else:
                        max_col = max(c for _, c in grid.keys()) if grid else -1
                        cell = _nearest_free_grid_cell(grid, 0, max_col + 1)
                else:
                    max_col = max(c for _, c in grid.keys()) if grid else -1
                    cell = _nearest_free_grid_cell(grid, 0, max_col + 1)

        grid[cell] = name
        name_to_grid[name] = cell

    # --- Convert grid → pixel coordinates (shared helper) ---
    edge_pairs = _build_edge_pairs(relationships, class_names)
    _grid_to_pixel_positions(
        grid, sizes, class_names, occupied, canvas_bounds,
        default_size=(CLASS_WIDTH, CLASS_MIN_HEIGHT),
        edge_pairs=edge_pairs, n_elements=n_classes,
    )

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

    Uses link-aware grid placement with dynamic canvas expansion and
    centering for large diagrams.
    """
    objects: List[Dict[str, Any]] = system_spec.get("objects", [])
    if not objects:
        return system_spec

    links: List[Dict[str, Any]] = system_spec.get("links", [])
    n_objects = len(objects)
    canvas_bounds = _dynamic_canvas_bounds(n_objects)

    occupied = extract_occupied_rects(existing_model, "ObjectDiagram")

    # --- Build adjacency graph from links ---
    obj_names: Dict[str, Dict[str, Any]] = {
        o.get("objectName", ""): o for o in objects
    }
    adjacency: Dict[str, Set[str]] = {name: set() for name in obj_names}
    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        if src in adjacency and tgt in adjacency:
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)

    # --- Compute sizes ---
    sizes: Dict[str, Tuple[int, int]] = {}
    for obj in objects:
        name = obj.get("objectName", "")
        sizes[name] = estimate_object_size(obj)

    # --- BFS placement order (most-connected first) ---
    placement_order: List[str] = []
    remaining = set(obj_names.keys())
    while remaining:
        root = max(remaining, key=lambda n: (len(adjacency.get(n, set())), n))
        queue = [root]
        bfs_seen: Set[str] = {root}
        while queue:
            current = queue.pop(0)
            placement_order.append(current)
            remaining.discard(current)
            for n in sorted(adjacency.get(current, set())):
                if n not in bfs_seen and n in remaining:
                    bfs_seen.add(n)
                    queue.append(n)

    # --- Assign grid cells ---
    ideal_rows, ideal_cols = _ideal_grid_shape(n_objects)
    grid: Dict[Tuple[int, int], str] = {}
    name_to_grid: Dict[str, Tuple[int, int]] = {}

    for name in placement_order:
        if not grid:
            cell = (0, 0)
        else:
            placed_neighbors = [
                (n, name_to_grid[n])
                for n in adjacency.get(name, set())
                if n in name_to_grid
            ]
            if placed_neighbors:
                cell = _best_neighbor_grid_cell(grid, placed_neighbors)
            else:
                max_col = max(c for _, c in grid.keys()) if grid else -1
                if max_col + 1 >= ideal_cols:
                    max_row = max(r for r, _ in grid.keys()) if grid else 0
                    cell = _nearest_free_grid_cell(grid, max_row + 1, 0)
                else:
                    cell = _nearest_free_grid_cell(grid, 0, max_col + 1)

        grid[cell] = name
        name_to_grid[name] = cell

    if not grid:
        return system_spec

    # --- Convert grid → pixel coordinates (shared helper) ---
    edge_pairs = _build_edge_pairs(links, obj_names)
    _grid_to_pixel_positions(
        grid, sizes, obj_names, occupied, canvas_bounds,
        default_size=(OBJECT_WIDTH, OBJECT_MIN_HEIGHT),
        edge_pairs=edge_pairs, n_elements=n_objects,
    )

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
    """Assign positions to a complete state machine using transition-aware layout.

    Layout strategy:
    - BFS from initial state following transitions to determine flow order
    - Transition-aware grid: connected states placed in adjacent cells
    - Initial state anchored at left, final state(s) at right
    - Dynamic canvas expansion for large state machines
    - Layout centered on origin for clean presentation
    """
    states: List[Dict[str, Any]] = system_spec.get("states", [])
    if not states:
        return system_spec

    transitions = system_spec.get("transitions", [])
    n_states = len(states)
    canvas_bounds = _dynamic_canvas_bounds(n_states)

    occupied = extract_occupied_rects(existing_model, "StateMachineDiagram")

    # --- Build state lookup and adjacency graph ---
    state_names: Dict[str, Dict[str, Any]] = {
        s.get("stateName", ""): s for s in states
    }
    adjacency: Dict[str, Set[str]] = {name: set() for name in state_names}
    outgoing: Dict[str, List[str]] = {name: [] for name in state_names}

    for t in transitions:
        src = t.get("source", "")
        tgt = t.get("target", "")
        if src in adjacency and tgt in adjacency:
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)
            outgoing[src].append(tgt)

    # --- Categorize states ---
    initial_states: List[str] = []
    final_states: List[str] = []
    regular_states: List[str] = []
    for s in states:
        name = s.get("stateName", "")
        stype = s.get("stateType", "regular")
        if stype == "initial":
            initial_states.append(name)
        elif stype == "final":
            final_states.append(name)
        else:
            regular_states.append(name)

    # --- BFS from initial state(s) to determine flow order ---
    visited: List[str] = []
    visited_set: Set[str] = set()
    queue: List[str] = []

    for name in initial_states:
        if name and name not in visited_set:
            queue.append(name)
            visited.append(name)
            visited_set.add(name)

    while queue:
        current = queue.pop(0)
        # Sort outgoing targets: regular states before final, then alphabetical
        targets = sorted(
            [tgt for tgt in outgoing.get(current, []) if tgt not in visited_set],
            key=lambda n: (
                0 if state_names.get(n, {}).get("stateType", "regular") == "regular" else 1,
                n,
            ),
        )
        for tgt in targets:
            visited.append(tgt)
            visited_set.add(tgt)
            queue.append(tgt)

    # Add unreachable states
    for name in regular_states + final_states:
        if name and name not in visited_set:
            visited.append(name)
            visited_set.add(name)

    # --- Compute element sizes ---
    sizes: Dict[str, Tuple[int, int]] = {}
    for s in states:
        name = s.get("stateName", "")
        sizes[name] = estimate_state_size(s)

    # --- Assign grid cells using transition-aware placement ---
    ideal_rows, ideal_cols = _ideal_grid_shape(n_states)
    grid: Dict[Tuple[int, int], str] = {}
    name_to_grid: Dict[str, Tuple[int, int]] = {}

    for name in visited:
        if not grid:
            cell = (0, 0)
        else:
            stype = state_names.get(name, {}).get("stateType", "regular")
            placed_neighbors = [
                (n, name_to_grid[n])
                for n in adjacency.get(name, set())
                if n in name_to_grid
            ]

            if stype == "final" and placed_neighbors:
                # Final states go to the right of their last connected state
                last_neighbor = placed_neighbors[-1]
                nr, nc = last_neighbor[1]
                cell = _nearest_free_grid_cell(grid, nr, nc + 1)
            elif placed_neighbors:
                cell = _best_neighbor_grid_cell(grid, placed_neighbors)
            else:
                # Isolated state — append in next available slot
                max_col = max(c for _, c in grid.keys()) if grid else -1
                if max_col + 1 >= ideal_cols:
                    # Wrap to next row
                    max_row = max(r for r, _ in grid.keys()) if grid else 0
                    cell = _nearest_free_grid_cell(grid, max_row + 1, 0)
                else:
                    cell = _nearest_free_grid_cell(grid, 0, max_col + 1)

        grid[cell] = name
        name_to_grid[name] = cell

    if not grid:
        return system_spec

    # --- Convert grid → pixel coordinates (shared helper) ---
    edge_pairs = _build_edge_pairs(transitions, state_names)
    _grid_to_pixel_positions(
        grid, sizes, state_names, occupied, canvas_bounds,
        default_size=(STATE_WIDTH, STATE_MIN_HEIGHT),
        edge_pairs=edge_pairs, n_elements=n_states,
    )

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

    Layout strategy:
    - Two-lane layout: intents in upper lane, states in lower lane
    - Initial node anchored at left, centered vertically between lanes
    - Dynamic canvas expansion for large agent diagrams
    - Grid-based placement within each lane with centering
    - Transition-aware: connected intents/states placed near each other
    """
    states_list: List[Dict[str, Any]] = system_spec.get("states", [])
    intents_list: List[Dict[str, Any]] = system_spec.get("intents", [])
    initial_nodes: List[Dict[str, Any]] = system_spec.get("initialNodes", [])

    n_total = len(states_list) + len(intents_list) + len(initial_nodes)
    canvas_bounds = _dynamic_canvas_bounds(n_total)
    c_min_x, c_max_x, c_min_y, c_max_y = canvas_bounds

    occupied = extract_occupied_rects(existing_model, "AgentDiagram")

    # --- Compute sizes for all elements ---
    intent_sizes: List[Tuple[int, int]] = []
    for intent in intents_list:
        intent_sizes.append(estimate_agent_element_size({"type": "intent", **intent}))

    state_sizes: List[Tuple[int, int]] = []
    for state in states_list:
        state_sizes.append(estimate_agent_element_size({"type": "state", **state}))

    # --- Calculate lane dimensions ---
    # Determine how many columns per lane
    n_intents = len(intents_list)
    n_states = len(states_list)
    _, intent_cols = _ideal_grid_shape(max(n_intents, 1))
    _, state_cols = _ideal_grid_shape(max(n_states, 1))

    h_gap = 60   # compact horizontal gap
    v_gap = 50   # compact vertical gap
    lane_gap = 60  # vertical gap between intent lane and state lane

    # --- Calculate intent lane dimensions ---
    intent_row_heights: List[int] = []
    intent_col_widths: List[int] = []
    if n_intents > 0:
        intent_rows_count = max(1, (n_intents + intent_cols - 1) // intent_cols)
        for row_idx in range(intent_rows_count):
            row_start = row_idx * intent_cols
            row_end = min(row_start + intent_cols, n_intents)
            row_max_h = max((intent_sizes[i][1] for i in range(row_start, row_end)), default=AGENT_NODE_MIN_HEIGHT)
            intent_row_heights.append(row_max_h)
        for col_idx in range(min(intent_cols, n_intents)):
            col_max_w = max(
                (intent_sizes[row_idx * intent_cols + col_idx][0]
                 for row_idx in range((n_intents + intent_cols - 1) // intent_cols)
                 if row_idx * intent_cols + col_idx < n_intents),
                default=AGENT_INTENT_WIDTH,
            )
            intent_col_widths.append(col_max_w)

    intent_lane_width = sum(intent_col_widths) + h_gap * max(0, len(intent_col_widths) - 1) if intent_col_widths else 0
    intent_lane_height = sum(intent_row_heights) + v_gap * max(0, len(intent_row_heights) - 1) if intent_row_heights else 0

    # --- Calculate state lane dimensions ---
    state_row_heights: List[int] = []
    state_col_widths: List[int] = []
    if n_states > 0:
        state_rows_count = max(1, (n_states + state_cols - 1) // state_cols)
        for row_idx in range(state_rows_count):
            row_start = row_idx * state_cols
            row_end = min(row_start + state_cols, n_states)
            row_max_h = max((state_sizes[i][1] for i in range(row_start, row_end)), default=AGENT_NODE_MIN_HEIGHT)
            state_row_heights.append(row_max_h)
        for col_idx in range(min(state_cols, n_states)):
            col_max_w = max(
                (state_sizes[row_idx * state_cols + col_idx][0]
                 for row_idx in range((n_states + state_cols - 1) // state_cols)
                 if row_idx * state_cols + col_idx < n_states),
                default=AGENT_STATE_WIDTH,
            )
            state_col_widths.append(col_max_w)

    state_lane_width = sum(state_col_widths) + h_gap * max(0, len(state_col_widths) - 1) if state_col_widths else 0
    state_lane_height = sum(state_row_heights) + v_gap * max(0, len(state_row_heights) - 1) if state_row_heights else 0

    # --- Compute total layout size and center ---
    initial_col_width = INITIAL_NODE_SIZE + h_gap if initial_nodes else 0
    total_width = max(intent_lane_width, state_lane_width) + initial_col_width
    total_height = intent_lane_height + lane_gap + state_lane_height

    origin_x = _snap(-total_width // 2)
    origin_y = _snap(-total_height // 2)

    # --- Place initial node(s) ---
    initial_x = origin_x
    initial_center_y = _snap(origin_y + total_height // 2 - INITIAL_NODE_SIZE // 2)
    for node in initial_nodes:
        w, h = INITIAL_NODE_SIZE, INITIAL_NODE_SIZE
        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=initial_x,
                                    preferred_y=initial_center_y,
                                    canvas_bounds=canvas_bounds)
        node["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))

    content_start_x = origin_x + initial_col_width

    # --- Place intents in upper lane (grid) ---
    intent_start_y = origin_y
    for idx, intent in enumerate(intents_list):
        row_idx = idx // intent_cols
        col_idx = idx % intent_cols
        w, h = intent_sizes[idx]

        px = content_start_x + sum(intent_col_widths[c] + h_gap for c in range(col_idx) if c < len(intent_col_widths))
        py = intent_start_y + sum(intent_row_heights[r] + v_gap for r in range(row_idx) if r < len(intent_row_heights))
        # Center horizontally within cell
        if col_idx < len(intent_col_widths):
            px += (intent_col_widths[col_idx] - w) // 2

        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=_snap(px),
                                    preferred_y=_snap(py),
                                    canvas_bounds=canvas_bounds)
        intent["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))

    # --- Place states in lower lane (grid) ---
    state_start_y = origin_y + intent_lane_height + lane_gap
    for idx, state in enumerate(states_list):
        row_idx = idx // state_cols
        col_idx = idx % state_cols
        w, h = state_sizes[idx]

        px = content_start_x + sum(state_col_widths[c] + h_gap for c in range(col_idx) if c < len(state_col_widths))
        py = state_start_y + sum(state_row_heights[r] + v_gap for r in range(row_idx) if r < len(state_row_heights))
        if col_idx < len(state_col_widths):
            px += (state_col_widths[col_idx] - w) // 2

        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=_snap(px),
                                    preferred_y=_snap(py),
                                    canvas_bounds=canvas_bounds)
        state["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))

    # --- Compute transition directions ---
    agent_transitions: List[Dict[str, Any]] = system_spec.get("transitions", [])
    all_agent_elements: Dict[str, Dict[str, Any]] = {}
    for node in initial_nodes:
        name = node.get("name", node.get("stateName", ""))
        if name:
            all_agent_elements[name] = node
    for intent in intents_list:
        name = intent.get("intentName", intent.get("name", ""))
        if name:
            all_agent_elements[name] = intent
    for state in states_list:
        name = state.get("stateName", state.get("name", ""))
        if name:
            all_agent_elements[name] = state

    _compute_edge_directions(
        agent_transitions,
        {name: (spec.get("position", {}), estimate_agent_element_size(spec))
         for name, spec in all_agent_elements.items()},
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
