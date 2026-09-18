"""Standalone A/B latency experiment for complete-system class-diagram generation.

MEASUREMENT ONLY — no product code is touched. This script replicates, as
faithfully as possible, the single structured-output LLM call that dominates a
"create a class diagram for X" turn:

    src/diagram_handlers/core/base_handler.py :: predict_structured()
      (reached via predict_two_pass_structured's fast path — every test domain
       here is < 250 chars, so the real handler also skips the reasoning pass
       and makes exactly ONE structured call)

Faithfulness notes (what is reused from the repo, via sys.path):
  * system prompt  : ClassDiagramHandler._get_system_generation_prompt() (verbatim)
  * user message   : f"User Request: {request}" (verbatim fast-path format)
  * schema         : schemas.class_diagram.SystemClassSpec (the real Pydantic model)
  * API surface    : client.beta.chat.completions.parse(...) (same as base_handler)
  * max tokens     : agent_config.LLM_MAX_TOKENS_LARGE (8192; SystemClassSpec is
                     not in _SMALL_OUTPUT_SCHEMAS, so the LARGE cap applies)
  * temperature vs reasoning_effort: decided by the repo's own
    model_config.supports_custom_temperature / reasoning_effort_for — gpt-5*
    models get reasoning_effort (default "low") and NO temperature; gpt-4o-mini
    gets temperature=LLM_TEMPERATURE (0.2) and no reasoning_effort. Identical
    to the branch in predict_structured.

Auth: the OpenAI key is read from <repo>/config.yaml (nlp.openai.api_key) or the
OPENAI_API_KEY env var — the same sources the deployed agent uses. The key is
NEVER printed, logged, or written anywhere.

Test matrix (<= 25 calls total; a config aborts after 2 errored calls):
  a) baseline        gpt-5-mini, exact current setup
  b) economy         baseline + appended be-economical instruction
  c) alt-model       gpt-4o-mini, baseline prompt/schema (temperature path)
  c2) minimal-effort gpt-5-mini with reasoning_effort="minimal" (the provider
                     path DOES pass reasoning_effort through, so this lever is
                     testable; note model_config comments that gpt-5.5 rejects
                     "minimal" — if gpt-5-mini rejects it too, the config
                     aborts after 2 errors and that is itself the finding)
  d) compact-text    plain PlantUML completion (chat.completions.create, no
                     structured output) — NOT product-ready (would need a parser)

Usage (run from anywhere; paths are absolute):
    python speed_experiment.py            # live run (requires an API key)
    python speed_experiment.py --dry-run  # build + print every request, 0 API calls
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\sulejmani\Desktop\BESSER-Experimental\modeling-agent")
OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"

sys.path.insert(0, str(REPO / "src"))

# --- Repo imports: reuse the product's own prompt / schema / model plumbing ---
from agent_config import LLM_TEMPERATURE, LLM_MAX_TOKENS_LARGE  # noqa: E402
from model_config import (  # noqa: E402
    reasoning_effort_for,
    supports_custom_temperature,
)
from schemas.class_diagram import SystemClassSpec  # noqa: E402
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler  # noqa: E402

MAX_TOTAL_CALLS = 25
MAX_ERRORS_PER_CONFIG = 2

# Baseline model: the parent investigation verified the deployed generation
# call runs on gpt-5-mini. NOTE: this repo's model_config.py DEFAULT for
# MODEL_GENERATION_LARGE is "gpt-5.6-terra" (env-overridable via
# BESSER_AGENT_MODEL_GENERATION_LARGE) — the deployment overrides it. The
# experiment pins the verified deployed model explicitly.
BASELINE_MODEL = "gpt-5-mini"
ALT_MODEL = "gpt-4o-mini"

ECONOMY_INSTRUCTION = (
    "\n\nBE ECONOMICAL: include only the classes the domain genuinely needs "
    "(5-7 for a typical domain). Do NOT add redundant getters/setters or "
    "filler methods. Prefer a few well-chosen attributes per class over "
    "exhaustive lists, and keep all names and descriptions terse."
)

PLANTUML_OVERRIDE = (
    "\n\nOUTPUT FORMAT OVERRIDE (this replaces any output format implied "
    "above): respond with ONLY a PlantUML class diagram, nothing else.\n"
    "- Start with @startuml and end with @enduml. No prose, no markdown fences.\n"
    "- One block per class:  class Name {\\n  type attrName\\n }\n"
    "- Enumerations:  enum Name {\\n  LITERAL\\n }\n"
    "- One line per relationship with multiplicities, e.g.:\n"
    '  Library "1" *-- "0..*" Book\n'
    '  Member "1" -- "0..*" Loan\n'
    "  SavingsAccount --|> Account\n"
)

DOMAINS = [
    {
        "id": "library",
        "request": "a library with books, members and loans",
        # Each inner list is a synonym group; a group is "hit" when any class
        # name contains any synonym (case-insensitive substring).
        "expected": [["book"], ["member", "user"], ["loan", "borrow"]],
    },
    {
        "id": "ecommerce",
        "request": "an e-commerce shop with products, orders and payments",
        "expected": [["product"], ["order"], ["payment"]],
    },
    {
        "id": "hospital",
        "request": "a hospital with patients, doctors and appointments",
        "expected": [["patient"], ["doctor", "physician"], ["appointment"]],
    },
    {
        "id": "hotel",
        "request": "a hotel booking system",
        "expected": [["room"], ["booking", "reservation"], ["guest", "customer", "user"]],
    },
    {
        "id": "tracker",
        "request": "a project tracker with teams, issues and sprints",
        "expected": [["team"], ["issue", "task", "ticket", "bug"], ["sprint"]],
    },
]


# ---------------------------------------------------------------------------
# Auth — key comes from config.yaml or env; NEVER printed or persisted.
# ---------------------------------------------------------------------------

def load_api_key() -> str | None:
    cfg_path = REPO / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml  # lazy: only needed when config.yaml exists
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            key = (((data.get("nlp") or {}).get("openai")) or {}).get("api_key")
            if key and isinstance(key, str) and key.strip():
                return key.strip()
        except Exception as exc:  # parse error etc. — fall through to env
            print(f"[warn] could not read config.yaml ({type(exc).__name__}); trying env")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


# ---------------------------------------------------------------------------
# Request builders — mirror base_handler.predict_structured / _predict_raw
# ---------------------------------------------------------------------------

def build_structured_kwargs(model: str, system_prompt: str, request: str,
                            reasoning_effort_override: str | None = None) -> dict:
    """Exactly the parse_kwargs branch of base_handler.predict_structured."""
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Request: {request}"},
        ],
        "response_format": SystemClassSpec,
        "max_completion_tokens": LLM_MAX_TOKENS_LARGE,
    }
    if supports_custom_temperature(model):
        kwargs["temperature"] = LLM_TEMPERATURE
    else:
        kwargs["reasoning_effort"] = (
            reasoning_effort_override or reasoning_effort_for(model)
        )
    return kwargs


def build_plain_kwargs(model: str, system_prompt: str, request: str) -> dict:
    """Mirrors base_handler._predict_raw's direct-client branch, but with the
    system prompt as a proper system message (the structured path does the
    same; keeping message roles identical isolates the format lever)."""
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Request: {request}"},
        ],
        "max_completion_tokens": LLM_MAX_TOKENS_LARGE,
    }
    if supports_custom_temperature(model):
        kwargs["temperature"] = LLM_TEMPERATURE
    else:
        kwargs["reasoning_effort"] = reasoning_effort_for(model)
    return kwargs


# ---------------------------------------------------------------------------
# Fidelity / shape metrics
# ---------------------------------------------------------------------------

def fidelity(class_names: list[str], expected_groups: list[list[str]]) -> tuple[int, int]:
    lowered = [c.lower() for c in class_names]
    hits = 0
    for group in expected_groups:
        if any(any(syn in name for name in lowered) for syn in group):
            hits += 1
    return hits, len(expected_groups)


_PUML_CLASS_RE = re.compile(
    r'^\s*(?:abstract\s+)?(?:class|enum|interface)\s+"?([A-Za-z_]\w*)"?',
    re.MULTILINE,
)
_PUML_REL_RE = re.compile(
    r'^\s*\S+.*?(?:--|\.\.|-+\|>|<\|-+|\*-+|o-+|-+>)\s*.*\S\s*$', re.MULTILINE,
)


def parse_plantuml(text: str) -> dict:
    """Cheap plausibility parse of a PlantUML response (config d)."""
    classes = _PUML_CLASS_RE.findall(text or "")
    # Relationship lines: contain a PlantUML connector and two endpoints, and
    # are not class/enum declaration lines.
    rel_lines = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith(("class ", "enum ", "interface ", "abstract ",
                                  "@", "'", "{", "}", "skinparam", "hide")):
            continue
        if re.search(r'(--|\.\.>|-+\|>|<\|-+|\*-+|o-+|-->)', s):
            rel_lines.append(s)
    plausible = (
        "@startuml" in (text or "")
        and "@enduml" in (text or "")
        and len(classes) >= 2
        and len(rel_lines) >= 1
    )
    return {
        "class_names": classes,
        "n_classes": len(classes),
        "n_relationships": len(rel_lines),
        "plausible_plantuml": plausible,
    }


def spec_metrics(spec: dict) -> dict:
    classes = spec.get("classes") or []
    names = [c.get("className", "") for c in classes if isinstance(c, dict)]
    return {
        "class_names": names,
        "n_classes": len(classes),
        "n_relationships": len(spec.get("relationships") or []),
        "n_methods_total": sum(len(c.get("methods") or []) for c in classes if isinstance(c, dict)),
        "n_attributes_total": sum(len(c.get("attributes") or []) for c in classes if isinstance(c, dict)),
    }


def usage_fields(completion) -> dict:
    u = getattr(completion, "usage", None)
    out = {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None}
    if u is not None:
        out["prompt_tokens"] = getattr(u, "prompt_tokens", None)
        out["completion_tokens"] = getattr(u, "completion_tokens", None)
        details = getattr(u, "completion_tokens_details", None)
        if details is not None:
            out["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return out


# ---------------------------------------------------------------------------
# The two call kinds
# ---------------------------------------------------------------------------

def run_structured_call(client, kwargs: dict) -> dict:
    t0 = time.perf_counter()
    completion = client.beta.chat.completions.parse(**kwargs)
    wall = time.perf_counter() - t0
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = getattr(completion.choices[0].message, "refusal", None)
        raise RuntimeError(f"empty structured output (refusal={refusal!r})")
    spec = parsed.model_dump()
    rec = {"wall_s": round(wall, 2),
           "finish_reason": completion.choices[0].finish_reason}
    rec.update(usage_fields(completion))
    rec.update(spec_metrics(spec))
    return rec


def run_plain_call(client, kwargs: dict) -> dict:
    t0 = time.perf_counter()
    completion = client.chat.completions.create(**kwargs)
    wall = time.perf_counter() - t0
    text = completion.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError("empty plain-text output")
    rec = {"wall_s": round(wall, 2),
           "finish_reason": completion.choices[0].finish_reason}
    rec.update(usage_fields(completion))
    rec.update(parse_plantuml(text))
    return rec


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

def make_configs() -> list[dict]:
    handler = ClassDiagramHandler(None)  # LLM arg unused for prompt access
    base_prompt = handler._get_system_generation_prompt()
    return [
        {
            "name": "baseline",
            "kind": "structured",
            "model": BASELINE_MODEL,
            "system_prompt": base_prompt,
            "reasoning_effort_override": None,
            "note": "exact current production call",
        },
        {
            "name": "economy",
            "kind": "structured",
            "model": BASELINE_MODEL,
            "system_prompt": base_prompt + ECONOMY_INSTRUCTION,
            "reasoning_effort_override": None,
            "note": "baseline + be-economical instruction",
        },
        {
            "name": "alt-model",
            "kind": "structured",
            "model": ALT_MODEL,
            "system_prompt": base_prompt,
            "reasoning_effort_override": None,
            "note": "baseline prompt/schema on gpt-4o-mini (temperature path)",
        },
        {
            "name": "minimal-effort",
            "kind": "structured",
            "model": BASELINE_MODEL,
            "system_prompt": base_prompt,
            "reasoning_effort_override": "minimal",
            "note": ("gpt-5-mini with reasoning_effort=minimal (baseline uses "
                     "'low'); may be rejected by the API — abort-on-2-errors "
                     "then IS the finding"),
        },
        {
            "name": "compact-text",
            "kind": "plain",
            "model": BASELINE_MODEL,
            "system_prompt": base_prompt + PLANTUML_OVERRIDE,
            "reasoning_effort_override": None,
            "note": "plain PlantUML completion — needs a parser to be product-ready",
        },
    ]


def build_kwargs_for(config: dict, request: str) -> dict:
    if config["kind"] == "structured":
        return build_structured_kwargs(
            config["model"], config["system_prompt"], request,
            reasoning_effort_override=config["reasoning_effort_override"],
        )
    return build_plain_kwargs(config["model"], config["system_prompt"], request)


def redacted_kwargs_preview(kwargs: dict) -> dict:
    """For logging: replace message bodies with lengths (they contain no
    secrets, but they are long); everything else verbatim."""
    preview = {k: v for k, v in kwargs.items() if k not in ("messages", "response_format")}
    preview["response_format"] = getattr(kwargs.get("response_format"), "__name__", None)
    preview["messages"] = [
        {"role": m["role"], "content_chars": len(m["content"])}
        for m in kwargs["messages"]
    ]
    return preview


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def summarize(calls: list[dict]) -> dict:
    summary: dict = {}
    for cfg in {c["config"] for c in calls}:
        rows = [c for c in calls if c["config"] == cfg]
        ok = [c for c in rows if c["ok"]]
        hits = sum(c.get("fidelity_hits", 0) for c in ok)
        expected = sum(c.get("fidelity_expected", 0) for c in ok)
        summary[cfg] = {
            "n_calls": len(rows),
            "n_ok": len(ok),
            "mean_wall_s": _mean([c.get("wall_s") for c in ok]),
            "mean_prompt_tokens": _mean([c.get("prompt_tokens") for c in ok]),
            "mean_completion_tokens": _mean([c.get("completion_tokens") for c in ok]),
            "mean_reasoning_tokens": _mean([c.get("reasoning_tokens") for c in ok]),
            "mean_classes": _mean([c.get("n_classes") for c in ok]),
            "mean_relationships": _mean([c.get("n_relationships") for c in ok]),
            "fidelity_hit_rate": round(hits / expected, 3) if expected else None,
            "errors": [c["error"] for c in rows if not c["ok"]],
        }
    return summary


def print_summary_table(summary: dict, config_order: list[str]) -> None:
    hdr = (f"{'config':<15} {'ok':>5} {'wall_s':>8} {'cmpl_tok':>9} "
           f"{'rsn_tok':>8} {'classes':>8} {'rels':>6} {'fidelity':>9} {'vs base':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    base_wall = (summary.get("baseline") or {}).get("mean_wall_s")
    for cfg in config_order:
        s = summary.get(cfg)
        if not s:
            continue
        wall = s["mean_wall_s"]
        speedup = (f"{base_wall / wall:.2f}x"
                   if base_wall and wall else "-")
        fid = (f"{s['fidelity_hit_rate']:.0%}"
               if s["fidelity_hit_rate"] is not None else "-")
        print(f"{cfg:<15} {s['n_ok']:>2}/{s['n_calls']:<2} "
              f"{wall if wall is not None else '-':>8} "
              f"{s['mean_completion_tokens'] if s['mean_completion_tokens'] is not None else '-':>9} "
              f"{s['mean_reasoning_tokens'] if s['mean_reasoning_tokens'] is not None else '-':>8} "
              f"{s['mean_classes'] if s['mean_classes'] is not None else '-':>8} "
              f"{s['mean_relationships'] if s['mean_relationships'] is not None else '-':>6} "
              f"{fid:>9} {speedup:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="build every request, print previews, make ZERO API calls")
    args = parser.parse_args()

    configs = make_configs()
    config_order = [c["name"] for c in configs]

    if args.dry_run:
        print("DRY RUN — no API calls will be made.\n")
        for cfg in configs:
            kwargs = build_kwargs_for(cfg, DOMAINS[0]["request"])
            print(f"[{cfg['name']}] ({cfg['kind']}) {cfg['note']}")
            print(f"  {json.dumps(redacted_kwargs_preview(kwargs))}")
        total = len(configs) * len(DOMAINS)
        print(f"\nPlanned: {len(configs)} configs x {len(DOMAINS)} domains = "
              f"{total} calls (budget {MAX_TOTAL_CALLS})")
        key = load_api_key()
        print(f"API key available: {'YES (redacted)' if key else 'NO — live run would abort'}")
        return 0

    key = load_api_key()
    if not key:
        print("ABORT: no API key found in "
              f"{REPO / 'config.yaml'} (nlp.openai.api_key) or OPENAI_API_KEY env. "
              "No LLM calls were made.")
        return 2

    from openai import OpenAI
    client = OpenAI(api_key=key)
    del key  # never keep it around longer than needed

    calls: list[dict] = []
    total_calls = 0

    for cfg in configs:
        errors = 0
        for domain in DOMAINS:
            if errors >= MAX_ERRORS_PER_CONFIG:
                print(f"[{cfg['name']}] aborted after {errors} errors — skipping remaining domains")
                break
            if total_calls >= MAX_TOTAL_CALLS:
                print("Global call budget reached — stopping.")
                break
            kwargs = build_kwargs_for(cfg, domain["request"])
            record: dict = {
                "config": cfg["name"], "kind": cfg["kind"], "model": cfg["model"],
                "domain": domain["id"], "request": domain["request"],
                "request_kwargs": redacted_kwargs_preview(kwargs),
            }
            total_calls += 1
            print(f"[{cfg['name']}] {domain['id']} ... ", end="", flush=True)
            try:
                if cfg["kind"] == "structured":
                    rec = run_structured_call(client, kwargs)
                else:
                    rec = run_plain_call(client, kwargs)
                hits, expected = fidelity(rec.get("class_names", []), domain["expected"])
                rec["fidelity_hits"] = hits
                rec["fidelity_expected"] = expected
                record.update(rec)
                record["ok"] = True
                record["error"] = None
                print(f"{rec['wall_s']}s, {rec.get('completion_tokens')} cmpl tok, "
                      f"{rec['n_classes']} classes, fidelity {hits}/{expected}")
            except Exception as exc:
                errors += 1
                record["ok"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"ERROR: {record['error']}")
            calls.append(record)

    summary = summarize(calls)
    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "repo": str(REPO),
            "baseline_model": BASELINE_MODEL,
            "alt_model": ALT_MODEL,
            "max_completion_tokens": LLM_MAX_TOKENS_LARGE,
            "temperature_when_supported": LLM_TEMPERATURE,
            "reasoning_effort_default": reasoning_effort_for(BASELINE_MODEL),
            "schema": "SystemClassSpec",
            "note": ("Replica of base_handler.predict_structured fast path; "
                     "api key sourced from config.yaml/env, never persisted."),
        },
        "calls": calls,
        "summary": summary,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRaw results written to {RESULTS_PATH}")
    print_summary_table(summary, config_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
