"""
User Profile Handler
Handles generation of BESSER User Profile models (UserDiagram).

A user-profile model describes a *target user* as a set of class-instance
boxes drawn from a fixed metamodel (User, Personal_Information, Competence,
Language, …).  Each attribute row is a matching *criterion* carrying a
comparison operator (``age >= 18``, ``level == B2``) rather than a plain
instance value.

This handler mirrors :class:`ObjectDiagramHandler`, with two differences:

1. The reference catalog is the **bundled metamodel** (the editor does not
   transmit it) — see :func:`utilities.user_metamodel.load_user_metamodel`.
2. Attributes carry an inferred ``operator``.
"""

import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from ..core.base_handler import BaseDiagramHandler, LLMPredictionError
from ..core.prompt_fragments import (
    EXACT_NAMES_RULE,
    MULTI_MOD_ARRAY_RULE,
    POSITION_DISCLAIMER,
    REMOVE_ELEMENT_RULE,
)
from schemas import (
    SystemUserProfileSpec,
    UserProfileModificationResponse,
)
from utilities.model_context import detailed_model_summary
from utilities.user_metamodel import load_user_metamodel

logger = logging.getLogger(__name__)

DIAGRAM_TYPE = "UserDiagram"
ROOT_CLASS = "User"

_OPERATORS = ("<", "<=", "==", ">=", ">")


MODIFY_SYSTEM_PROMPT_USER = f"""You are a user-profile modeling expert. The user wants to modify a user-profile model.

A user-profile model is a set of class-instance boxes (drawn from a fixed metamodel) whose attribute rows are matching CRITERIA with a comparison operator (e.g. "age >= 18", "level == B2").

IMPORTANT RULES:
1. Actions available: "add_object", "modify_object", "modify_attribute_value", "add_link", "remove_element"
2. add_object: set target.profileName to a short lowercase instance identifier (e.g. "personal_information1", "language2"). Put className, classId, and attributes in "changes". Each attribute needs name, operator (one of {", ".join(_OPERATORS)}), and value. FILL IN EVERY attribute of the class with a plausible, coherent value inferred from the request (e.g. a disability "paraplegia" -> name == "Paraplegia", description == "Paralysis of the lower limbs", affects == a valid aspect literal). Leave only uniquely-identifying attributes (firstName, lastName, address) blank.
3. modify_attribute_value: set target.profileName + target.attributeName; put the new operator and/or value in "changes".
4. {EXACT_NAMES_RULE}
5. {REMOVE_ELEMENT_RULE}
6. {MULTI_MOD_ARRAY_RULE}
7. Use ONLY classes and attributes from the metamodel below; copy classId and attributeId verbatim — the frontend uses these ids to link the profile back to the metamodel."""


class UserProfileDiagramHandler(BaseDiagramHandler):
    """Handler for User Profile (UserDiagram) generation."""

    def get_diagram_type(self) -> str:
        return DIAGRAM_TYPE

    # ------------------------------------------------------------------
    # Reference catalog (bundled metamodel)
    # ------------------------------------------------------------------

    def _reference(self) -> Dict[str, Any]:
        """Return the bundled metamodel as a class-diagram-shaped dict."""
        return load_user_metamodel()

    def _sanitize_profile_name(self, value: str, default_name: str = "profile1") -> str:
        if not isinstance(value, str):
            return default_name
        base = re.sub(r"[^A-Za-z0-9_]", "", value.strip())
        if not base:
            return default_name
        if not base[0].isalpha():
            base = f"p{base}"
        return base[0].lower() + base[1:]

    def _normalize_operator(self, value: Any) -> str:
        if not isinstance(value, str):
            return "=="
        op = value.strip()
        if op == "=":
            return "=="
        return op if op in _OPERATORS else "=="

    def _extract_reference_catalog(
        self, reference_diagram: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
        """Build {className.lower(): {name,id,attributes[]}} and a relationship list.

        Also caches enumeration literals on ``self._enum_literals``.
        """
        if not isinstance(reference_diagram, dict):
            return {}, []

        elements = reference_diagram.get("elements")
        relationships = reference_diagram.get("relationships")
        if not isinstance(elements, dict):
            return {}, []

        # First pass: collect enumerations and their literals.
        self._enum_literals: Dict[str, List[str]] = {}
        for element in elements.values():
            if not isinstance(element, dict) or element.get("type") != "Enumeration":
                continue
            enum_name = (element.get("name") or "").strip()
            if not enum_name:
                continue
            literals = []
            for attr_id in element.get("attributes", []):
                attr = elements.get(attr_id)
                if isinstance(attr, dict):
                    lit = (attr.get("name") or "").replace("+ ", "").replace("- ", "").replace("# ", "").split(":")[0].strip()
                    if lit:
                        literals.append(lit)
            self._enum_literals[enum_name] = literals

        classes: Dict[str, Dict[str, Any]] = {}
        by_id: Dict[str, Dict[str, Any]] = {}

        for class_id, element in elements.items():
            if not isinstance(element, dict):
                continue
            if element.get("type") not in ("Class", "AbstractClass"):
                continue
            class_name = element.get("name")
            if not isinstance(class_name, str) or not class_name.strip():
                continue
            class_name = class_name.strip()
            class_attrs: List[Dict[str, str]] = []
            for attr_id in element.get("attributes", []):
                attr = elements.get(attr_id)
                if not isinstance(attr, dict) or attr.get("type") != "ClassAttribute":
                    continue
                raw_name = str(attr.get("name", "")).replace("+ ", "").replace("- ", "").replace("# ", "")
                attr_name = raw_name.split(":")[0].strip()
                if not attr_name:
                    continue
                class_attrs.append(
                    {
                        "name": attr_name,
                        "id": attr_id,
                        "type": str(attr.get("attributeType", "str")),
                    }
                )

            info = {"name": class_name, "id": class_id, "attributes": class_attrs}
            classes[class_name.lower()] = info
            by_id[class_id] = info

        class_relationships: List[Dict[str, str]] = []
        if isinstance(relationships, dict):
            for relation in relationships.values():
                if not isinstance(relation, dict):
                    continue
                source = relation.get("source")
                target = relation.get("target")
                if not isinstance(source, dict) or not isinstance(target, dict):
                    continue
                src_id = source.get("element")
                tgt_id = target.get("element")
                if src_id not in by_id or tgt_id not in by_id:
                    continue
                rel_name = relation.get("name")
                if not isinstance(rel_name, str) or not rel_name.strip():
                    rel_name = "relatedTo"
                class_relationships.append(
                    {
                        "sourceClass": by_id[src_id]["name"],
                        "targetClass": by_id[tgt_id]["name"],
                        "name": rel_name.strip(),
                    }
                )

        return classes, class_relationships

    def _format_reference_classes(self, elements: Dict[str, Any]) -> str:
        """Format metamodel classes + enums for the LLM prompt."""
        formatted: List[str] = []

        enum_literals: Dict[str, List[str]] = {}
        for el in elements.values():
            if not isinstance(el, dict) or el.get("type") != "Enumeration":
                continue
            enum_name = (el.get("name") or "").strip()
            if not enum_name:
                continue
            literals: List[str] = []
            for attr_id in el.get("attributes", []):
                attr = elements.get(attr_id)
                if isinstance(attr, dict):
                    lit = (attr.get("name") or "").replace("+ ", "").replace("- ", "").replace("# ", "").split(":")[0].strip()
                    if lit:
                        literals.append(lit)
            enum_literals[enum_name] = literals
        self._enum_literals = enum_literals

        for enum_name, literals in enum_literals.items():
            if literals:
                formatted.append(f"\nEnumeration: {enum_name} — valid values: {', '.join(literals)}")

        classes = {k: v for k, v in elements.items() if isinstance(v, dict) and v.get("type") in ("Class", "AbstractClass")}
        for class_id, class_data in classes.items():
            class_name = class_data.get("name", "Unknown")
            formatted.append(f"\nClass: {class_name} (classId: {class_id})")
            formatted.append("Attributes:")
            for attr_id in class_data.get("attributes", []):
                if attr_id not in elements:
                    continue
                attr = elements[attr_id]
                attr_name = str(attr.get("name", "")).replace("+ ", "").replace("- ", "").replace("# ", "").split(":")[0].strip()
                attr_type = attr.get("attributeType", "str")
                type_info = f", type: {attr_type}" if attr_type else ""
                if attr_type in enum_literals:
                    type_info += f" [valid values: {', '.join(enum_literals[attr_type])}]"
                formatted.append(f"  - {attr_name} (attributeId: {attr_id}{type_info})")

        return "\n".join(formatted)

    def _format_reference_relationships(self, relationships: List[Dict[str, str]]) -> str:
        if not relationships:
            return "No explicit metamodel relationships were found."
        return "\n".join(
            f"- {rel['sourceClass']} -> {rel['targetClass']} (name: {rel['name']})"
            for rel in relationships
        )

    # ------------------------------------------------------------------
    # Metamodel association graph (rooted at User)
    # ------------------------------------------------------------------

    def _association_graph(self) -> Dict[str, Dict[str, Any]]:
        """Return a parent-pointer tree of the metamodel rooted at ``User``.

        ``{child_class: {"parent": str, "assoc": str, "many": bool}}`` where
        ``many`` reflects the cardinality on the child end (``0..*`` / ``1..*``).
        Inheritance edges are ignored; only class-to-class associations whose
        both endpoints are real classes in the (short) metamodel are used.
        Cached because the metamodel is fixed.
        """
        cached = getattr(self, "_assoc_graph_cache", None)
        if cached is not None:
            return cached

        reference = self._reference()
        elements = reference.get("elements", {})
        relationships = reference.get("relationships", {})
        id2class = {
            i: e.get("name")
            for i, e in elements.items()
            if isinstance(e, dict) and e.get("type") in ("Class", "AbstractClass") and e.get("name")
        }

        # Undirected adjacency: class -> [(neighbor, neighbor_multiplicity, assoc_name)]
        adj: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for rel in relationships.values():
            if not isinstance(rel, dict):
                continue
            if "Inheritance" in (rel.get("type") or ""):
                continue
            source = rel.get("source") or {}
            target = rel.get("target") or {}
            sc = id2class.get(source.get("element"))
            tc = id2class.get(target.get("element"))
            if not sc or not tc or sc == tc:
                continue
            name = (rel.get("name") or "").strip() or "relatedTo"
            s_mult = str(source.get("multiplicity", ""))
            t_mult = str(target.get("multiplicity", ""))
            adj[sc].append((tc, t_mult, name))
            adj[tc].append((sc, s_mult, name))

        parent_of: Dict[str, Dict[str, Any]] = {}
        if ROOT_CLASS in id2class.values():
            visited = {ROOT_CLASS}
            queue: deque = deque([ROOT_CLASS])
            while queue:
                cur = queue.popleft()
                for neighbor, neighbor_mult, assoc in sorted(adj.get(cur, [])):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    parent_of[neighbor] = {
                        "parent": cur,
                        "assoc": assoc,
                        "many": "*" in neighbor_mult,
                    }
                    queue.append(neighbor)

        self._assoc_graph_cache = parent_of
        return parent_of

    def _class_icon(self, class_id: Optional[str], elements: Dict[str, Any]) -> Optional[str]:
        if not class_id:
            return None
        el = elements.get(class_id) if isinstance(elements, dict) else None
        if isinstance(el, dict) and isinstance(el.get("icon"), str):
            return el["icon"]
        return None

    def _is_singleton(self, cls: str) -> bool:
        """A class is a singleton box when its parent-end cardinality is at most 1."""
        if cls == ROOT_CLASS:
            return True
        info = self._association_graph().get(cls)
        return True if info is None else not info["many"]

    def _path_to_root(self, cls: str) -> List[str]:
        """Return the metamodel ancestry ``[User, ..., cls]`` (``[cls]`` if isolated)."""
        parent_of = self._association_graph()
        path = [cls]
        cursor, guard = cls, 0
        while cursor in parent_of and guard < 10:
            cursor = parent_of[cursor]["parent"]
            path.append(cursor)
            guard += 1
        path.reverse()
        return path

    @staticmethod
    def _model_class_index(
        current_model: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, str], set]:
        """Index an existing UserDiagram model.

        Returns ``(existing_by_class, existing_links)`` where
        ``existing_by_class`` maps a lowercased className to the box's display
        name, and ``existing_links`` is a set of ``(sourceClassLower,
        targetClassLower)`` pairs already connected.
        """
        existing_by_class: Dict[str, str] = {}
        existing_links: set = set()
        if not isinstance(current_model, dict):
            return existing_by_class, existing_links
        elements = current_model.get("elements")
        if not isinstance(elements, dict):
            return existing_by_class, existing_links

        for el in elements.values():
            if isinstance(el, dict) and el.get("type") == "UserModelName":
                cls = (el.get("className") or "").strip().lower()
                if cls:
                    existing_by_class.setdefault(cls, el.get("name") or cls)

        relationships = current_model.get("relationships")
        if isinstance(relationships, dict):
            for rel in relationships.values():
                if not isinstance(rel, dict):
                    continue
                s = elements.get((rel.get("source") or {}).get("element"))
                t = elements.get((rel.get("target") or {}).get("element"))
                if isinstance(s, dict) and isinstance(t, dict):
                    sc = (s.get("className") or "").strip().lower()
                    tc = (t.get("className") or "").strip().lower()
                    if sc and tc:
                        existing_links.add((sc, tc))
        return existing_by_class, existing_links

    def _structural_add_object(
        self,
        mod: Dict[str, Any],
        reference_classes: Dict[str, Dict[str, Any]],
        elements: Dict[str, Any],
        existing_by_class: Dict[str, str],
        existing_links: set,
        ensured: set,
    ) -> List[Dict[str, Any]]:
        """Expand a single ``add_object`` into a metamodel-faithful sequence.

        Ensures the box's whole ancestry up to ``User`` exists (creating only
        missing singleton boxes), then links each parent→child pair that isn't
        already connected. Existing boxes are referenced by className so the
        frontend modifier reuses them instead of duplicating.
        """
        changes = mod.get("changes") or {}
        class_name_raw = (changes.get("className") or "").strip()
        info = reference_classes.get(class_name_raw.lower())
        if not info:
            return [mod]  # unknown class — leave untouched

        class_name = info["name"]
        changes["className"] = class_name
        if not changes.get("classId"):
            changes["classId"] = info["id"]
        changes["attributes"] = self._full_attributes(info, changes.get("attributes"))
        # Carry the metamodel class icon so the frontend modifier can build the
        # UserModelIcon child (User diagrams always render in icon view).
        if not changes.get("icon"):
            changes["icon"] = self._class_icon(info["id"], elements)
        mod["changes"] = changes

        path = self._path_to_root(class_name)
        ref: Dict[str, str] = {c: c for c in path}  # link by className (resolves new+existing singletons)
        out: List[Dict[str, Any]] = []

        # 1) Ensure ancestor singleton boxes exist.
        for ancestor in path[:-1]:
            anc_l = ancestor.lower()
            if anc_l in ensured:
                continue
            anc_info = reference_classes.get(anc_l)
            out.append({
                "action": "add_object",
                "target": {"profileName": ancestor.lower()},
                "changes": {
                    "className": anc_info["name"] if anc_info else ancestor,
                    "classId": anc_info["id"] if anc_info else None,
                    "attributes": self._full_attributes(anc_info, []) if anc_info else [],
                    "icon": self._class_icon(anc_info["id"], elements) if anc_info else None,
                },
            })
            ensured.add(anc_l)

        # 2) The target box itself.
        leaf_l = class_name.lower()
        if self._is_singleton(class_name) and leaf_l in ensured:
            # Singleton already present — reuse it, don't add a duplicate box.
            ref[class_name] = class_name
        else:
            leaf_name = self._sanitize_profile_name(
                str((mod.get("target") or {}).get("profileName")
                    or changes.get("profileName")
                    or f"{class_name[0].lower()}{class_name[1:]}")
            )
            mod.setdefault("target", {})["profileName"] = leaf_name
            ref[class_name] = leaf_name
            out.append(mod)
            ensured.add(leaf_l)

        # 3) Link each parent→child pair that isn't already connected.
        for i in range(len(path) - 1):
            parent, child = path[i], path[i + 1]
            pair = (parent.lower(), child.lower())
            child_preexisting = child.lower() in existing_by_class
            already = pair in existing_links or (child.lower(), parent.lower()) in existing_links
            if already and child_preexisting:
                continue
            out.append({
                "action": "add_link",
                "target": {"sourceProfile": ref[parent], "targetProfile": ref[child]},
                "changes": {"source": ref[parent], "target": ref[child], "relationshipType": ""},
            })
            existing_links.add(pair)

        return out

    def _empty_box(self, class_name: str, classes: Dict[str, Dict[str, Any]],
                   elements: Dict[str, Any]) -> Dict[str, Any]:
        info = classes.get(class_name.lower())
        class_id = info["id"] if info else None
        return {
            "profileName": "",
            "className": info["name"] if info else class_name,
            "classId": class_id,
            "icon": self._class_icon(class_id, elements),
            # Auto-created boxes still carry their full (empty) attribute set.
            "attributes": self._full_attributes(info, []) if info else [],
        }

    @staticmethod
    def _merge_attrs(box: Dict[str, Any], attrs: List[Dict[str, Any]]) -> None:
        """Merge criteria from another box of the same class into ``box``.

        Both boxes carry the full attribute set, so for matching attribute
        names we only fill a value/operator when the target's is still empty;
        genuinely new attribute names are appended.
        """
        by_name = {a.get("name"): a for a in box.get("attributes", [])}
        for attr in attrs:
            target = by_name.get(attr.get("name"))
            if target is None:
                box.setdefault("attributes", []).append(attr)
                by_name[attr.get("name")] = attr
            elif not target.get("value") and attr.get("value"):
                target["value"] = attr["value"]
                target["operator"] = attr.get("operator", target.get("operator", "=="))

    def _assemble_structure(
        self,
        enriched_boxes: List[Dict[str, Any]],
        classes: Dict[str, Dict[str, Any]],
        elements: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """Build a metamodel-faithful set of boxes + links rooted at ``User``.

        - Every model gets a single ``User`` root box.
        - ``0..1`` classes (Personal_Information, Competence, Accessibility, …)
          become at most one shared box; ``0..*`` classes (Language, Skill,
          Education, Disability) may have several.
        - Required intermediate boxes are auto-created (e.g. a Language box
          forces a Competence box) and every box is linked to its metamodel
          parent, all the way up to ``User``.
        """
        parent_of = self._association_graph()

        def is_singleton(cls: str) -> bool:
            if cls == ROOT_CLASS:
                return True
            info = parent_of.get(cls)
            return True if info is None else not info["many"]

        singleton_box: Dict[str, Dict[str, Any]] = {}
        multi_boxes: List[Dict[str, Any]] = []

        def ensure_singleton(cls: str) -> Dict[str, Any]:
            if cls not in singleton_box:
                singleton_box[cls] = self._empty_box(cls, classes, elements)
            return singleton_box[cls]

        # Place the LLM-produced boxes.
        for box in enriched_boxes:
            cls = box["className"]
            if is_singleton(cls):
                if cls in singleton_box:
                    self._merge_attrs(singleton_box[cls], box.get("attributes", []))
                else:
                    singleton_box[cls] = box
            else:
                multi_boxes.append(box)

        # Always include the User root, then ensure every ancestor box exists.
        ensure_singleton(ROOT_CLASS)
        present = set(singleton_box.keys()) | {b["className"] for b in multi_boxes}
        for cls in list(present):
            cursor = cls
            guard = 0
            while cursor in parent_of and guard < 10:
                parent = parent_of[cursor]["parent"]
                ensure_singleton(parent)
                cursor = parent
                guard += 1

        # Assign unique profile names.
        used: set = set()

        def unique(name: str) -> str:
            base = self._sanitize_profile_name(name)
            candidate, suffix = base, 2
            while candidate in used:
                candidate = f"{base}{suffix}"
                suffix += 1
            used.add(candidate)
            return candidate

        final: List[Dict[str, Any]] = []
        singleton_name: Dict[str, str] = {}
        for cls, box in singleton_box.items():
            box["profileName"] = unique("user_1" if cls == ROOT_CLASS else cls.lower())
            singleton_name[cls] = box["profileName"]
            final.append(box)
        for index, box in enumerate(multi_boxes, start=1):
            cls = box["className"]
            box["profileName"] = unique(f"{cls[0].lower()}{cls[1:]}{index}")
            final.append(box)

        # Build links along metamodel associations (parent box -> child box).
        links: List[Dict[str, str]] = []

        def add_link(child_cls: str, child_name: str) -> None:
            parent_name = (
                singleton_name.get(parent_of[child_cls]["parent"])
                if child_cls in parent_of
                else singleton_name.get(ROOT_CLASS)
            )
            if parent_name and parent_name != child_name:
                # Metamodel association names (e.g. "Competence_User_non_navigable")
                # are internal identifiers; the editor's own fixtures leave the
                # link label blank, so we do too.
                links.append({"source": parent_name, "target": child_name,
                              "relationshipType": ""})

        for cls, box in singleton_box.items():
            if cls == ROOT_CLASS:
                continue
            add_link(cls, box["profileName"])
        for box in multi_boxes:
            add_link(box["className"], box["profileName"])

        return final, links

    # ------------------------------------------------------------------
    # Catalog-driven normalization / enrichment
    # ------------------------------------------------------------------

    def _enrich_profile(self, profile: Dict[str, Any], classes: Dict[str, Dict[str, Any]],
                        elements: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Fill classId / attributeId / type from the catalog; drop unknown classes/attrs.

        Returns the normalized profile dict, or ``None`` when the className is
        not part of the metamodel (so callers can skip it).
        """
        class_name_raw = profile.get("className")
        if not isinstance(class_name_raw, str) or not class_name_raw.strip():
            return None
        info = classes.get(class_name_raw.strip().lower())
        if not info:
            logger.warning(f"[UserDiagram] Dropping profile with unknown class {class_name_raw!r}")
            return None

        class_name = info["name"]
        profile_name = self._sanitize_profile_name(
            str(profile.get("profileName") or f"{class_name}{index}"),
            default_name=f"{class_name[0].lower()}{class_name[1:]}{index}",
        )

        return {
            "profileName": profile_name,
            "className": class_name,
            "classId": info["id"],
            "icon": self._class_icon(info["id"], elements),
            # Metamodel attributes are NOT optional: emit every attribute the
            # class defines, filling in the criteria the user gave and leaving
            # the rest blank (the editor shows them as empty rows to complete).
            "attributes": self._full_attributes(info, profile.get("attributes")),
        }

    def _full_attributes(
        self, class_info: Dict[str, Any], provided: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Return the class's complete attribute list, merging provided criteria.

        Every metamodel attribute of the class appears (in metamodel order).
        Where the request supplied a matching criterion its operator/value are
        used; otherwise the row defaults to ``==`` with an empty value.
        """
        by_name: Dict[str, Dict[str, Any]] = {}
        for attr in provided or []:
            if isinstance(attr, dict) and isinstance(attr.get("name"), str):
                by_name[attr["name"].strip().lower()] = attr

        attributes: List[Dict[str, Any]] = []
        for ref in class_info.get("attributes", []):
            given = by_name.get(ref["name"].lower(), {})
            attributes.append(
                {
                    "name": ref["name"],
                    "attributeId": ref["id"],
                    "operator": self._normalize_operator(given.get("operator")),
                    "value": str(given.get("value", "")) if given else "",
                    "type": ref.get("type", "str"),
                }
            )
        return attributes

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return f"""You are a user-profile modeling expert. Create a SINGLE user-profile class-instance box based on the user's request.

A user-profile box instantiates a class from the metamodel and lists matching CRITERIA as attribute rows. Each criterion has a comparison operator.

CRITICAL RULES:
1. Use ONLY classes and attributes that exist in the METAMODEL below.
2. className MUST match a metamodel class exactly. classId/attributeId are filled in automatically — you may omit them.
3. ENUMERATIONS — STRICT: if an attribute's type matches a listed enumeration, the value MUST be one of that enumeration's valid literals. Never invent enum values.
4. Each attribute MUST have: name (exact metamodel attribute name), operator (one of {", ".join(_OPERATORS)}), value.
5. Infer the operator from phrasing: "older than 18" -> '>', "at least B2" -> '>=', "under 18" -> '<', otherwise '=='.
6. {POSITION_DISCLAIMER}"""

    def _system_prompt_for_system(self) -> str:
        return f"""You are a user-profile modeling expert. Describe a target user by choosing which METAMODEL classes hold the relevant matching CRITERIA.

Each box instantiates a metamodel class and lists criteria as attribute rows with comparison operators (e.g. "age >= 18", "level == B2").

IMPORTANT RULES:
1. Use ONLY classes/attributes from the METAMODEL below. Put each criterion on the class that actually owns the attribute:
   - age / firstName / lastName / gender / nationality / address -> Personal_Information
   - a language and its proficiency (iso693_3, level) -> Language
   - a skill (name, score) -> Skill
   - religion -> Culture
   - a disability (name, description, affects) -> Disability
   - degree / field of study -> Education
2. Do NOT output a "User" box, intermediate boxes (e.g. Competence, Accessibility), or any links — the root User element, the required intermediate boxes, and all connections are added automatically from the metamodel associations. Just emit the leaf boxes that carry criteria.
3. A profile may contain several boxes of the same class (e.g. multiple Language boxes for multiple languages).
4. FILL IN EVERY attribute of each class you include — don't just set the one the user named. Infer plausible, coherent values for the rest from the persona. Examples:
   - "speaks Portuguese" (native/only language) -> Language: iso693_3 == "por", level == "C2"
   - "has sight issues" -> Disability: affects == "Sight", name == "Visual impairment", description == "Reduced or impaired vision"
   EXCEPTION: leave attributes that uniquely identify ONE individual (firstName, lastName, address) blank (empty value) — a profile represents a GROUP of users, so those wouldn't generalize. Always fill descriptive/characterizing attributes.
5. Each attribute MUST have name (exact metamodel attribute), operator (one of {", ".join(_OPERATORS)}), and value.
6. Infer operators from phrasing: "older than 18" -> '>', "at least B2" -> '>=', "under 18" -> '<', otherwise '=='.
7. ENUMERATIONS — STRICT: enum-typed values MUST be one of the listed valid literals.
8. classId/attributeId/profileName are filled in automatically — you may omit them.
9. {POSITION_DISCLAIMER}"""

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None,
                                **kwargs) -> Dict[str, Any]:
        # A user profile is only meaningful when rooted at User with the right
        # metamodel links, so even a "single" request produces a complete,
        # structured profile rather than an isolated box.
        return self.generate_complete_system(user_request, existing_model, **kwargs)

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None,
                                 **kwargs) -> Dict[str, Any]:
        reference = self._reference()
        elements = reference.get("elements", {})
        classes, class_relationships = self._extract_reference_catalog(reference)

        system_prompt = self._system_prompt_for_system()
        user_prompt = (
            f"{user_request}\n\n"
            f"METAMODEL (use these exact classes and attributes):\n"
            f"{self._format_reference_classes(elements)}\n\n"
            f"METAMODEL RELATIONSHIPS:\n"
            f"{self._format_reference_relationships(class_relationships)}"
        )

        try:
            parsed = self.predict_structured(user_prompt, SystemUserProfileSpec, system_prompt=system_prompt)
            spec = parsed.model_dump()
            system_spec = self._normalize_system(spec, classes, elements, class_relationships)

            for profile in system_spec.get("profiles", []):
                profile.pop("position", None)
            self.apply_system_layout(system_spec, existing_model)

            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": DIAGRAM_TYPE,
                "message": self._build_system_message(system_spec),
            }
        except LLMPredictionError:
            logger.error("[UserDiagram] generate_complete_system LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that user profile. Please try again or rephrase.")
        except Exception:
            logger.error("[UserDiagram] generate_complete_system FAILED", exc_info=True)
            return self.generate_fallback_system()

    def _normalize_system(self, spec: Dict[str, Any], classes: Dict[str, Dict[str, Any]],
                          elements: Dict[str, Any], relationships: List[Dict[str, str]]) -> Dict[str, Any]:
        """Resolve LLM classes/attributes, then wire a metamodel-faithful tree.

        The LLM only chooses which metamodel classes hold the criteria; the
        ``User`` root, the required intermediate boxes, and every link are
        derived deterministically from the metamodel associations — the LLM's
        own ``links`` are ignored.
        """
        raw_profiles = spec.get("profiles") if isinstance(spec.get("profiles"), list) else []

        enriched: List[Dict[str, Any]] = []
        for index, raw in enumerate(raw_profiles, start=1):
            if not isinstance(raw, dict):
                continue
            box = self._enrich_profile(raw, classes, elements, index)
            if box:
                enriched.append(box)

        profiles, links = self._assemble_structure(enriched, classes, elements)

        system_name = spec.get("systemName")
        if not isinstance(system_name, str) or not system_name.strip():
            system_name = "UserProfile"

        return {
            "systemName": system_name.strip(),
            "profiles": profiles,
            "links": links,
        }

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_system_spec(self) -> Dict[str, Any]:
        """A minimal but metamodel-faithful starter: User -> Personal_Information."""
        reference = self._reference()
        elements = reference.get("elements", {})
        classes, _ = self._extract_reference_catalog(reference)
        seed = self._empty_box("Personal_Information", classes, elements)
        profiles, links = self._assemble_structure([seed], classes, elements)
        return {"systemName": "UserProfile", "profiles": profiles, "links": links}

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        return self.generate_fallback_system()

    def generate_fallback_system(self) -> Dict[str, Any]:
        spec = self._fallback_system_spec()
        self.apply_system_layout(spec)
        return {
            "action": "inject_complete_system",
            "systemSpec": spec,
            "diagramType": DIAGRAM_TYPE,
            "message": (
                "I created a starter user profile. Describe your target user in more detail "
                "(e.g. *'a teenager who speaks Spanish at B2 and is studying engineering'*) "
                "and I'll build a richer profile!"
            ),
        }

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _attr_preview(attr: Dict[str, Any]) -> str:
        op = attr.get("operator", "==")
        display = "=" if op == "==" else op
        return f'`{attr.get("name", "")} {display} {attr.get("value", "")}`'

    def _build_single_message(self, spec: Dict[str, Any]) -> str:
        cls = spec.get("className", "Profile")
        attrs = spec.get("attributes", [])
        msg = f"Created a **{cls}** profile box"
        if attrs:
            preview = [self._attr_preview(a) for a in attrs[:4]]
            msg += f" with criteria: {', '.join(preview)}"
            if len(attrs) > 4:
                msg += f" (+{len(attrs) - 4} more)"
        msg += ". You can ask me to add more boxes or refine the criteria!"
        return msg

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        system_name = spec.get("systemName", "UserProfile")
        profiles = spec.get("profiles", [])
        links = spec.get("links", [])
        names = [p.get("className", "?") for p in profiles[:6]]
        msg = f"Built the **{system_name}** user profile with {len(profiles)} box(es)"
        if names:
            msg += f": {', '.join(f'**{n}**' for n in names)}"
            if len(profiles) > 6:
                msg += f" (+{len(profiles) - 6} more)"
        if links:
            msg += f" and {len(links)} link(s)"
        msg += ". Feel free to ask me to refine criteria or add more boxes!"
        return msg

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None,
                              **kwargs) -> Dict[str, Any]:
        reference = self._reference()
        elements = reference.get("elements", {})
        reference_classes, _ = self._extract_reference_catalog(reference)

        context_block = ""
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, DIAGRAM_TYPE)
            if summary:
                context_block = f"\n\n{summary}"

        reference_context = "\n\nMETAMODEL (use these classes and attributes):\n" + self._format_reference_classes(elements)
        user_prompt = f"Modify the user-profile model: {user_request}{context_block}{reference_context}"

        logger.info(f"[UserDiagram] generate_modification called with: {user_request!r}")

        existing_by_class, existing_links = self._model_class_index(current_model)

        def _resolve_class_references(mod_list: list) -> list:
            """Expand add_object mods so the new box is wired into the metamodel.

            Each add_object gains its full attribute set and is preceded by any
            missing ancestor boxes (e.g. Accessibility for a Disability) plus
            the links connecting it up to User. Existing boxes are reused.
            """
            ensured = set(existing_by_class.keys())
            out: list = []
            for mod in mod_list:
                if isinstance(mod, dict) and mod.get("action") == "add_object":
                    out.extend(self._structural_add_object(
                        mod, reference_classes, elements,
                        existing_by_class, existing_links, ensured,
                    ))
                else:
                    out.append(mod)
            return out

        try:
            return self._execute_modification(
                user_prompt, MODIFY_SYSTEM_PROMPT_USER, UserProfileModificationResponse,
                post_processor=_resolve_class_references,
            )
        except LLMPredictionError as exc:
            logger.error(f"[UserDiagram] generate_modification LLM FAILED: {exc}")
            return self._error_response("I couldn't process that modification. Please try again or rephrase.")
        except Exception as exc:
            logger.error(f"[UserDiagram] generate_modification FAILED: {exc}", exc_info=True)
            return {
                "action": "modify_model",
                "modification": {
                    "action": "modify_object",
                    "target": {"profileName": "Unknown"},
                    "changes": {"profileName": "ModifiedProfile"},
                },
                "diagramType": DIAGRAM_TYPE,
                "message": (
                    "I couldn't apply that modification automatically. Could you rephrase it? "
                    "For example: *'set the age criterion to >= 21'* or *'add a Language box for French at C1'*."
                ),
            }

    def _build_mod_target_name(self, action: str, target: dict, mod: dict = None) -> str:
        """Resolve user-profile target names for friendly modification messages."""
        name = (
            target.get("profileName")
            or target.get("className")
            or target.get("attributeName")
        )
        attr_name = target.get("attributeName")
        profile = target.get("profileName") or target.get("className")
        if profile and attr_name and action in ("remove_element", "modify_attribute_value"):
            return f"attribute {attr_name} on {profile}"
        return name or "profile"
