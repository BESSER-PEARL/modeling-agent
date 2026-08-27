"""Addon config for the speed experiment: COMPACT JSON (the user's idea).

Same gpt-5-mini brain and structured-output enforcement as the baseline, but a
minimal schema: 1-letter keys and string-encoded members ("price: float",
"decreasePrice(pct: float) -> None") instead of one JSON object per attribute.
Keeps the parse guarantee PlantUML free-text lacks, with (hopefully) most of
its token savings. Runs ONLY the new config (5 calls) and compares against the
baseline mean from results.json.

Run inside the agent container:  python scratch_speed_experiment/compact_json_addon.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Literal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import BaseModel, Field  # noqa: E402

from speed_experiment import (  # noqa: E402  (reuse the verified harness plumbing)
    BASELINE_MODEL, DOMAINS, LLM_MAX_TOKENS_LARGE, fidelity, load_api_key,
    supports_custom_temperature, reasoning_effort_for, usage_fields,
)
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler  # noqa: E402


# ---------------------------------------------------------------------------
# Compact schema — the experiment's subject
# ---------------------------------------------------------------------------

class CClass(BaseModel):
    n: str = Field(..., description="class name, PascalCase")
    a: List[str] = Field(..., description=(
        "attributes, each encoded as 'name: type' (types: str, int, float, "
        "bool, datetime, date, time, or an enumeration name)"))
    m: List[str] = Field(..., description=(
        "methods, each encoded as 'name(param: type, ...) -> returnType'; "
        "omit '-> ...' when it returns nothing; '' params when none"))


class CEnum(BaseModel):
    n: str = Field(..., description="enumeration name, PascalCase")
    v: List[str] = Field(..., description="literal values, UPPER_CASE")


class CRel(BaseModel):
    f: str = Field(..., description="source class name")
    t: str = Field(..., description="target class name")
    k: Literal["association", "composition", "aggregation", "inheritance"] = (
        Field(..., description="relationship kind"))
    c: str = Field(..., description=(
        "cardinality 'sourceMult..targetMult' e.g. '1..*', '*..1', '1..1'; "
        "'' for inheritance"))
    l: str = Field(..., description="relationship name/label; '' if none")


class CompactSpec(BaseModel):
    name: str = Field(..., description="system name, PascalCase")
    classes: List[CClass]
    enums: List[CEnum]
    rels: List[CRel]


COMPACT_ADDENDUM = (
    "\n\nOUTPUT FORMAT OVERRIDE: reply via the provided COMPACT schema. All "
    "the modeling rules above still apply (same completeness: every entity "
    "the domain needs, attributes with types, meaningful methods, every "
    "relationship with sensible cardinalities) — only the ENCODING is "
    "compact: attributes as 'name: type' strings, methods as "
    "'name(param: type, ...) -> returnType' strings, relationships as "
    "{f, t, k, c, l}. Do not add prose."
)


# ---------------------------------------------------------------------------
# Metrics for the compact shape
# ---------------------------------------------------------------------------

def compact_metrics(spec: dict) -> dict:
    classes = spec.get("classes") or []
    names = [c.get("n", "") for c in classes if isinstance(c, dict)]
    n_attrs = sum(len(c.get("a") or []) for c in classes if isinstance(c, dict))
    n_methods = sum(len(c.get("m") or []) for c in classes if isinstance(c, dict))
    # sanity: are the string encodings well-formed?
    bad_attrs = sum(
        1 for c in classes if isinstance(c, dict)
        for s in (c.get("a") or []) if ":" not in s
    )
    bad_methods = sum(
        1 for c in classes if isinstance(c, dict)
        for s in (c.get("m") or []) if "(" not in s
    )
    return {
        "class_names": names,
        "n_classes": len(classes),
        "n_relationships": len(spec.get("rels") or []),
        "n_attributes_total": n_attrs,
        "n_methods_total": n_methods,
        "n_enums": len(spec.get("enums") or []),
        "malformed_attr_strings": bad_attrs,
        "malformed_method_strings": bad_methods,
    }


def main() -> int:
    api_key = load_api_key()
    if not api_key:
        print("No API key available; aborting.")
        return 1
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    handler = ClassDiagramHandler(None)
    system_prompt = handler._get_system_generation_prompt() + COMPACT_ADDENDUM

    calls = []
    errors = 0
    for domain in DOMAINS:
        if errors >= 2:
            print("aborting config after 2 errors")
            break
        request = f"create a class diagram for {domain['request']}"
        kwargs: dict = {
            "model": BASELINE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Request: {request}"},
            ],
            "response_format": CompactSpec,
            "max_completion_tokens": LLM_MAX_TOKENS_LARGE,
        }
        if supports_custom_temperature(BASELINE_MODEL):
            kwargs["temperature"] = 0.2
        else:
            kwargs["reasoning_effort"] = reasoning_effort_for(BASELINE_MODEL)
        try:
            t0 = time.perf_counter()
            completion = client.beta.chat.completions.parse(**kwargs)
            wall = time.perf_counter() - t0
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("empty structured output")
            spec = parsed.model_dump()
            rec = {"config": "compact-json", "domain": domain["id"], "ok": True,
                   "wall_s": round(wall, 2),
                   "finish_reason": completion.choices[0].finish_reason}
            rec.update(usage_fields(completion))
            rec.update(compact_metrics(spec))
            hit, total = fidelity(rec["class_names"], domain["expected"])
            rec["fidelity"] = f"{hit}/{total}"
            rec["fidelity_ok"] = hit == total
            calls.append(rec)
            print(f"[compact-json] {domain['id']} ... {rec['wall_s']}s, "
                  f"{rec['completion_tokens']} cmpl tok, {rec['n_classes']} classes, "
                  f"{rec['n_relationships']} rels, fidelity {rec['fidelity']}, "
                  f"malformed a/m: {rec['malformed_attr_strings']}/{rec['malformed_method_strings']}")
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[compact-json] {domain['id']} ERROR: {exc}")
            calls.append({"config": "compact-json", "domain": domain["id"],
                          "ok": False, "error": str(exc)})

    ok_calls = [c for c in calls if c.get("ok")]
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_compact_json.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calls, f, indent=2)
    print(f"\nRaw results written to {out_path}")

    if ok_calls:
        mean = lambda k: sum(c[k] for c in ok_calls) / len(ok_calls)  # noqa: E731
        # baseline mean from the main run's results.json (same directory)
        base_wall = None
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results.json"), encoding="utf-8") as f:
                base = [c for c in json.load(f)["calls"]
                        if c.get("config") == "baseline" and c.get("ok")]
            if base:
                base_wall = sum(c["wall_s"] for c in base) / len(base)
        except Exception:
            pass
        print(f"\ncompact-json: {len(ok_calls)}/5 ok | mean wall "
              f"{mean('wall_s'):.2f}s | mean cmpl tok {mean('completion_tokens'):.0f} | "
              f"mean classes {mean('n_classes'):.1f} | mean rels {mean('n_relationships'):.1f} | "
              f"fidelity {sum(1 for c in ok_calls if c['fidelity_ok'])}/{len(ok_calls)} | "
              f"malformed strings {sum(c['malformed_attr_strings'] + c['malformed_method_strings'] for c in ok_calls)}")
        if base_wall:
            print(f"vs baseline ({base_wall:.2f}s): {base_wall / mean('wall_s'):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
