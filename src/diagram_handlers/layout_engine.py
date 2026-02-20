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

CANVAS_MIN_X = -900
CANVAS_MAX_X = 900
CANVAS_MIN_Y = -500
CANVAS_MAX_Y = 500

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
) -> Tuple[int, int]:
    """Find the first non-overlapping position using a scanning strategy.

    Starts near (preferred_x, preferred_y) and scans outward.
    """
    step_x = width + H_GAP
    step_y = height + V_GAP

    # Try the preferred position first
    candidate = Rect(_snap(preferred_x), _snap(preferred_y), width, height)
    if not _collides(candidate, occupied):
        return candidate.x, candidate.y

    # Spiral outward from the preferred position
    for ring in range(1, 40):
        for dx_mult in range(-ring, ring + 1):
            for dy_mult in range(-ring, ring + 1):
                if abs(dx_mult) != ring and abs(dy_mult) != ring:
                    continue  # only check the ring perimeter
                cx = _snap(preferred_x + dx_mult * step_x)
                cy = _snap(preferred_y + dy_mult * step_y)
                # Keep within canvas bounds
                if cx < CANVAS_MIN_X or cx + width > CANVAS_MAX_X:
                    continue
                if cy < CANVAS_MIN_Y or cy + height > CANVAS_MAX_Y:
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
    """
    classes: List[Dict[str, Any]] = system_spec.get("classes", [])
    relationships: List[Dict[str, Any]] = system_spec.get("relationships", [])
    if not classes:
        return system_spec

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
                # Isolated class — extend rightward in first row
                max_col = max(c for _, c in grid.keys()) if grid else -1
                cell = _nearest_free_grid_cell(grid, 0, max_col + 1)

        grid[cell] = name
        name_to_grid[name] = cell

    # --- Convert grid → pixel coordinates ---
    if not grid:
        return system_spec

    min_row = min(r for r, _ in grid.keys())
    min_col = min(c for _, c in grid.keys())
    max_row = max(r for r, _ in grid.keys())
    max_col = max(c for _, c in grid.keys())

    col_widths: Dict[int, int] = {}
    for col in range(min_col, max_col + 1):
        col_widths[col] = max(
            (sizes.get(grid.get((row, col), ""), (CLASS_WIDTH, CLASS_MIN_HEIGHT))[0]
             for row in range(min_row, max_row + 1) if (row, col) in grid),
            default=CLASS_WIDTH,
        )

    row_heights: Dict[int, int] = {}
    for row in range(min_row, max_row + 1):
        row_heights[row] = max(
            (sizes.get(grid.get((row, col), ""), (CLASS_WIDTH, CLASS_MIN_HEIGHT))[1]
             for col in range(min_col, max_col + 1) if (row, col) in grid),
            default=CLASS_MIN_HEIGHT,
        )

    # --- Relationship-aware gaps between adjacent columns / rows ---
    # Build a symmetric set of class-name pairs that share a relationship
    rel_pairs: Set[Tuple[str, str]] = set()
    for rel in relationships:
        s = rel.get("source", "")
        t = rel.get("target", "")
        if s in class_names and t in class_names:
            rel_pairs.add((s, t))
            rel_pairs.add((t, s))

    col_gaps: Dict[int, int] = {}
    for col in range(min_col, max_col):
        names_a = [grid[(r, col)] for r in range(min_row, max_row + 1)
                   if (r, col) in grid]
        names_b = [grid[(r, col + 1)] for r in range(min_row, max_row + 1)
                   if (r, col + 1) in grid]
        has_rel = any((a, b) in rel_pairs for a in names_a for b in names_b)
        col_gaps[col] = H_GAP + REL_EXTRA_GAP if has_rel else H_GAP

    row_gaps: Dict[int, int] = {}
    for row in range(min_row, max_row):
        names_a = [grid[(row, c)] for c in range(min_col, max_col + 1)
                   if (row, c) in grid]
        names_b = [grid[(row + 1, c)] for c in range(min_col, max_col + 1)
                   if (row + 1, c) in grid]
        has_rel = any((a, b) in rel_pairs for a in names_a for b in names_b)
        row_gaps[row] = V_GAP + REL_EXTRA_GAP if has_rel else V_GAP

    start_x = _snap(CANVAS_MIN_X + 100)
    start_y = _snap(CANVAS_MIN_Y + 60)

    for (row, col), name in grid.items():
        w, h = sizes.get(name, (CLASS_WIDTH, CLASS_MIN_HEIGHT))
        # Pixel = start + sum of preceding column widths + per-gap spacing
        px = start_x + sum(col_widths.get(c, CLASS_WIDTH) + col_gaps.get(c, H_GAP)
                           for c in range(min_col, col))
        py = start_y + sum(row_heights.get(r, CLASS_MIN_HEIGHT) + row_gaps.get(r, V_GAP)
                           for r in range(min_row, row))
        # Centre horizontally within cell; top-align vertically
        cell_w = col_widths.get(col, CLASS_WIDTH)
        px += (cell_w - w) // 2

        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=_snap(px),
                                    preferred_y=_snap(py))
        class_names[name]["position"] = {"x": x, "y": y}
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
    """Assign positions to all objects in a complete object diagram."""
    objects: List[Dict[str, Any]] = system_spec.get("objects", [])
    if not objects:
        return system_spec

    occupied = extract_occupied_rects(existing_model, "ObjectDiagram")
    start_x = _snap(CANVAS_MIN_X + 100)
    start_y = _snap(CANVAS_MIN_Y + 60)
    row_x = start_x
    row_y = start_y
    row_max_height = 0

    for obj in objects:
        w, h = estimate_object_size(obj)
        if row_x + w > CANVAS_MAX_X - 100 and row_x != start_x:
            row_x = start_x
            row_y += row_max_height + V_GAP
            row_max_height = 0
        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=row_x, preferred_y=row_y)
        obj["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))
        row_x = x + w + H_GAP
        row_max_height = max(row_max_height, h)

    # --- Compute link directions ---
    links: List[Dict[str, Any]] = system_spec.get("links", [])
    obj_names: Dict[str, Dict[str, Any]] = {
        o.get("objectName", ""): o for o in objects
    }
    _compute_edge_directions(
        links,
        {name: (spec.get("position", {}), estimate_object_size(spec))
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
    """Assign positions to a complete state machine (left-to-right flow).

    Places initial state on the left, regular states flowing rightward,
    and final state on the right.
    """
    states: List[Dict[str, Any]] = system_spec.get("states", [])
    if not states:
        return system_spec

    occupied = extract_occupied_rects(existing_model, "StateMachineDiagram")

    # Sort: initial first, then regular, then final
    type_order = {"initial": 0, "regular": 1, "final": 2}
    ordered = sorted(states, key=lambda s: type_order.get(s.get("stateType", "regular"), 1))

    # Build a transition adjacency to order regular states by flow
    transitions = system_spec.get("transitions", [])
    state_names = {s.get("stateName", ""): s for s in states}
    # BFS from initial state
    visited: List[str] = []
    queue: List[str] = []
    for s in ordered:
        if s.get("stateType") == "initial":
            name = s.get("stateName", "")
            if name:
                queue.append(name)
                visited.append(name)
                break
    while queue:
        current = queue.pop(0)
        for t in transitions:
            if t.get("source") == current:
                target = t.get("target", "")
                if target and target not in visited:
                    visited.append(target)
                    queue.append(target)
    # Add any states not reachable from initial
    for s in ordered:
        name = s.get("stateName", "")
        if name and name not in visited:
            visited.append(name)

    # Place states left-to-right following the BFS order
    start_x = _snap(CANVAS_MIN_X + 80)
    start_y = _snap((CANVAS_MIN_Y + CANVAS_MAX_Y) // 2 - STATE_MIN_HEIGHT // 2)
    cursor_x = start_x
    row_y = start_y

    for name in visited:
        s = state_names.get(name)
        if not s:
            continue
        w, h = estimate_state_size(s)
        if cursor_x + w > CANVAS_MAX_X - 80:
            cursor_x = start_x
            row_y += h + V_GAP + 40
        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=cursor_x, preferred_y=row_y)
        s["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))
        cursor_x = x + w + H_GAP

    # --- Compute transition directions ---
    _compute_edge_directions(
        transitions,
        {name: (spec.get("position", {}), estimate_state_size(spec))
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

    Places intents in an upper lane and states in a lower lane,
    with the initial node on the left.
    """
    states_list: List[Dict[str, Any]] = system_spec.get("states", [])
    intents_list: List[Dict[str, Any]] = system_spec.get("intents", [])
    initial_nodes: List[Dict[str, Any]] = system_spec.get("initialNodes", [])

    occupied = extract_occupied_rects(existing_model, "AgentDiagram")

    # Place initial node(s) first, top-left
    cursor_x = _snap(CANVAS_MIN_X + 80)
    initial_y = _snap(CANVAS_MIN_Y + 40)
    for node in initial_nodes:
        w, h = INITIAL_NODE_SIZE, INITIAL_NODE_SIZE
        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=cursor_x, preferred_y=initial_y)
        node["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))
        cursor_x = x + w + H_GAP

    # Place intents in upper lane
    intent_y = _snap(CANVAS_MIN_Y + 120)
    intent_x = _snap(CANVAS_MIN_X + 100)
    for intent in intents_list:
        w, h = estimate_agent_element_size({"type": "intent", **intent})
        if intent_x + w > CANVAS_MAX_X - 80:
            intent_x = _snap(CANVAS_MIN_X + 100)
            intent_y += h + V_GAP
        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=intent_x, preferred_y=intent_y)
        intent["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))
        intent_x = x + w + H_GAP

    # Place states in lower lane
    state_y = _snap((CANVAS_MIN_Y + CANVAS_MAX_Y) // 2 + 40)
    state_x = _snap(CANVAS_MIN_X + 100)
    for state in states_list:
        w, h = estimate_agent_element_size({"type": "state", **state})
        if state_x + w > CANVAS_MAX_X - 80:
            state_x = _snap(CANVAS_MIN_X + 100)
            state_y += h + V_GAP
        x, y = _find_free_position(w, h, occupied,
                                    preferred_x=state_x, preferred_y=state_y)
        state["position"] = {"x": x, "y": y}
        occupied.append(Rect(x, y, w, h))
        state_x = x + w + H_GAP

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
