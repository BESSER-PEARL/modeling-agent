"""Which end of a composition or aggregation is the whole.

The convention, shared with the editor and BESSER (BESSER-PEARL/BESSER#746):
the SOURCE of a Composition or Aggregation is the PART and the TARGET is the
WHOLE. The editor draws the diamond at the target end, and BESSER's JSON ->
BUML conversion marks the target end ``is_composite``. It reads like an
Inheritance, whose target is the parent: "BrandURL -> Brand" is "a BrandURL
is part of a Brand".

A composite end is single-valued in UML (a part belongs to at most one whole
at a time), so a composition whose target end has an upper bound above 1
while its source end has one of at most 1 names its ends the wrong way round.
"""

from typing import Optional

WHOLE_PART_TYPES = frozenset({"composition", "aggregation"})


def whole_part_kind(rel_type: object) -> Optional[str]:
    """``"composition"`` / ``"aggregation"`` for either spelling, else None.

    Accepts the agent's names ("Composition") and the editor's
    ("ClassComposition").
    """
    if not isinstance(rel_type, str):
        return None
    key = rel_type.strip().lower()
    if key.startswith("class"):
        key = key[len("class"):]
    return key if key in WHOLE_PART_TYPES else None


def upper_bound(multiplicity: object) -> Optional[float]:
    """The upper bound of ``1``, ``0..1``, ``0..*``, ``*``, ``2..5``; None if unreadable."""
    if multiplicity is None:
        return None
    text = str(multiplicity).strip().replace(" ", "")
    if not text:
        return None
    upper = text.split("..")[-1]
    if upper in ("*", "n", "many"):
        return float("inf")
    try:
        return float(int(upper))
    except ValueError:
        return None


def is_reversed_composition(source_multiplicity: object, target_multiplicity: object) -> bool:
    """True when a composition's whole is evidently at its SOURCE end.

    The target (whole) end allows several instances while the source end
    allows at most one, which only fits if the ends are swapped. Anything
    else (both single-valued, a missing or unreadable multiplicity) is
    left alone: there is no evidence either way.
    """
    target_upper = upper_bound(target_multiplicity)
    source_upper = upper_bound(source_multiplicity)
    if target_upper is None or source_upper is None:
        return False
    return target_upper > 1 and source_upper <= 1


# Spec keys that belong to one end, as (source key, target key) pairs.
_END_KEY_PAIRS = (
    ("source", "target"),
    ("sourceMultiplicity", "targetMultiplicity"),
    ("sourceRole", "name"),
    ("sourceDirection", "targetDirection"),
)


def orient_composition_spec(rel: object) -> bool:
    """Swap the ends of a spec Composition stated with its whole as the SOURCE.

    *rel* is a relationship of a system spec (``source``, ``target``,
    ``sourceMultiplicity``, ``targetMultiplicity``, ``name`` = the target
    end's role, ``sourceRole``). Every per-end field moves with its end, so
    the roles and multiplicities keep their meaning. Returns True when it
    swapped. A composition whose ends are both single-valued, or an
    Aggregation (a shared whole may be many), is left alone.
    """
    if not isinstance(rel, dict) or whole_part_kind(rel.get("type")) != "composition":
        return False
    if not is_reversed_composition(rel.get("sourceMultiplicity"), rel.get("targetMultiplicity")):
        return False
    for src_key, tgt_key in _END_KEY_PAIRS:
        src_val, tgt_val = rel.get(src_key), rel.get(tgt_key)
        for key, val in ((src_key, tgt_val), (tgt_key, src_val)):
            if val is None:
                rel.pop(key, None)
            else:
                rel[key] = val
    return True
