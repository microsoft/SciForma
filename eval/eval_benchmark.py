#!/usr/bin/env python3
# =============================================================================
# SciForma Benchmark Evaluation Script
#
# OFFICIAL EVALUATION SETTINGS (hard-coded, do not modify):
#
#   split_dims = True          Each sample makes 3 independent GPT calls
#                              (one per axis: Component / Arrow / Text).
#                              Without this, Arrow scores are inflated ~+0.25.
#
#   rubrics_suffix = ""        Loads benchmark_{level}_rubrics.json.
#                              This matches the rubric inventory used in the
#                              original paper evaluation (April 2026).
#
# Standard usage:
#   python eval_benchmark.py \
#       --gen_dir    /path/to/model_outputs \
#       --output_dir /path/to/results \
#       [--llm_model gpt-5.4] \
#       [--workers 4] [--num_retest 2]
#
# Reference scores on SciFormaBench-2K (Simple 500 / Medium 900 / Hard 600):
#   SciForma-Base  67.59%  (Comp 73.52  Arrow 64.64  Text 63.84)  ← paper
#   SciForma-9B    69.51%  (Comp 74.49  Arrow 66.46  Text 67.00)  ← paper
#   GPT-Image-1.5  68.96%
#
# Reproduction (2026-06-26, gpt-5.4_2026-03-05): 68.59% and 68.65% on two runs
#   (±1% from paper due to gpt-5.4 version drift March→April 2026)
# =============================================================================
"""
Benchmark Evaluation — V9r rubrics-based scoring using GPT-5.4 (Azure OpenAI)
with round-robin multi-endpoint strategy.

Works with the benchmark_final/ directory structure:
  CLEAN mode (default):
    benchmark_simple.json  —  {validation_prompts, resolution_list, gt_images}
    benchmark_simple_rubrics.json  —  {total, successful, entries[{index, image_path, stage1_abstraction}]}
  INTERNAL mode (--internal):
    benchmark_final_from2k4/benchmark_simple.json  —  {validation_prompts, resolution_list, metadata}
    benchmark_final_from2k4/benchmark_simple_rubrics.json  —  {entries[{global_idx, image_path, stage1_abstraction}]}

Supports FOUR gen-image naming conventions (auto-detected per level):
  (A) NNNNN.png / NNNNN.jpg  (numeric)
  (B) promptNNNN_*.png  (e.g. prompt0000_A_neural_network.png)
  (C) prompt_NNNN/sample_0.png  (directory per prompt)
  (D) Fallback: any image with a numeric prefix

Round-robin strategy:
  - N Azure OpenAI clients (one per endpoint, ManagedIdentity or CLI auth).
  - Distributes requests cyclically across all healthy clients.
  - Retries on 429 / 5xx with exponential backoff and automatic failover.

Usage:
    python eval/eval_benchmark.py \\
        --gen_dir /data/benchmark_sample/my_model \\
        --output_dir ./eval_results/my_model_gpt54 \\
        --workers 16

    # Internal mode (with globalIdx metadata)
    python eval/eval_benchmark.py \\
        --gen_dir /data/benchmark_sample/my_model \\
        --internal \\
        --output_dir ./eval_results/my_model_gpt54

    # AMLT auth
    python eval/eval_benchmark.py \\
        --gen_dir /data/benchmark_sample/my_model \\
        --auth managed_identity
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Azure Endpoint Configuration
# ═══════════════════════════════════════════════════════════════════════

# ─── Load judge configuration from judge_config.py ────────────────────────────
# Copy eval/judge_config.example.py → eval/judge_config.py and fill in
# your credentials. judge_config.py is gitignored and never committed.
_JUDGE_BACKEND      = "azure_cli"   # default (overridden by judge_config if present)
_JUDGE_DEPLOYMENT   = "gpt-4o"     # paper used gpt-5.4; gpt-4o is the public equivalent
ENDPOINT_TOKEN_SCOPE: dict[str, str] = {}   # populated below from judge_config

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(__file__))
    import judge_config as _jc
    _JUDGE_BACKEND = getattr(_jc, "BACKEND", _JUDGE_BACKEND)
    if _JUDGE_BACKEND == "azure_cli":
        ENDPOINT_TOKEN_SCOPE = getattr(_jc, "AZURE_CLI_ENDPOINTS", {})
        _JUDGE_DEPLOYMENT    = getattr(_jc, "AZURE_CLI_DEPLOYMENT", _JUDGE_DEPLOYMENT)
    elif _JUDGE_BACKEND == "azure_apikey":
        # API-key endpoints stored as {endpoint: key} — handled in ClientPool
        ENDPOINT_TOKEN_SCOPE = getattr(_jc, "AZURE_APIKEY_ENDPOINTS", {})
        _JUDGE_DEPLOYMENT    = getattr(_jc, "AZURE_APIKEY_DEPLOYMENT", _JUDGE_DEPLOYMENT)
    # "openai" backend handled separately in ClientPool
except ImportError:
    pass   # no judge_config.py — fall back to env vars

# ── Env var fallback (no judge_config.py needed) ──────────────────────────────
# AZURE_OPENAI_ENDPOINTS: comma-separated list for round-robin (faster eval)
# AZURE_OPENAI_ENDPOINT:  single endpoint (backward compat)
import re as _re
_scope = "https://cognitiveservices.azure.com/.default"
_eps_raw = os.environ.get("AZURE_OPENAI_ENDPOINTS", "") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
if _eps_raw and not ENDPOINT_TOKEN_SCOPE:
    for _ep in [e.strip() for e in _eps_raw.split(",") if e.strip()]:
        _base_ep = _re.sub(r"/openai.*$", "", _ep.rstrip("/"))
        ENDPOINT_TOKEN_SCOPE[_base_ep + "/openai/v1"] = _scope
    _JUDGE_BACKEND = "cli"

AZURE_CLIENT_ID = ""   # set in judge_config.py if using managed identity
MAX_RETRIES = 8
RETRY_CODES = {429, 500, 502, 503, 504}


# ═══════════════════════════════════════════════════════════════════════
#  Scoring Constants & Prompts
# ═══════════════════════════════════════════════════════════════════════

SEVERITY_WEIGHTS: Dict[str, float] = {"critical": 1.0, "moderate": 0.5, "minor": 0.0}
MIN_EXPECTED: int = 1
MAX_EXPECTED_CAP: int = 30

STAGE2_SYSTEM = (
    "You are a scientific diagram quality evaluator. You will see TWO images "
    "and a structured element list extracted from the reference. "
    "LEFT = REFERENCE (ground truth). RIGHT = GENERATED (to evaluate). "
    "Your job is to identify clear, confident errors in the GENERATED image. "
    "Do NOT report minor stylistic differences or speculative issues."
)

STAGE2_USER_TEMPLATE = """## Context
You are evaluating a GENERATED scientific diagram produced by a diffusion model.
The model usually reproduces the overall layout and style well, but frequently
struggles with structural fidelity — missing components, wrong arrow directions,
garbled text labels. Focus on these confident structural errors.

The diagram should depict: "{prompt}"

LEFT image = REFERENCE. RIGHT image = GENERATED.

## Reference Element List (extracted from reference)
### Components ({n_comp} total)
{comp_list}

### Arrows / Connections ({n_arrow} total)
{arrow_list}

### Text Labels ({n_text} total)
{text_list}

## Your Task
Compare the GENERATED (right) image against the REFERENCE (left) image.
Report only errors you are **confident** about.
Do NOT guess or speculate — if you are unsure, leave it out.

**component_errors** — Components / blocks / shapes:
  - "Missing": a reference component is entirely absent → critical
  - "Wrong": present but depicts a clearly different concept → critical
  - "Structural mismatch": the component exists at roughly the right location,
    but its internal structure differs significantly from the reference
    (e.g. reference shows stacked tiles but generated shows a single flat block;
    reference shows a grid with cells but generated shows a plain rectangle;
    reference shows sub-elements inside a component but generated merges
    them into a featureless blob; dimension annotations like "M×N" on a
    matrix are missing) → critical
  - "Distorted": component is present but its shape is severely distorted,
    blurred, or illegible compared to the reference → critical
  - "Duplicate": appears more times than in reference → moderate
  Check not just whether each component is present, but whether its visual
  structure matches the reference. Do NOT report extra decorative elements
  or minor style differences (color, shadow, rounded vs sharp corners).
  Report ONLY the types listed above. Do NOT invent other categories.

**arrow_errors** — Arrows / connections:
  - "Missing": a reference connection is entirely absent → critical
  Arrow direction, endpoints, and routing cannot be reliably judged from
  rasterized diagram images. Therefore:
  Do NOT report reversed arrows, wrong endpoints, routing differences,
  curvature, thickness, color, or any directional errors.
  Do NOT report spurious/extra arrows.
  ONLY report arrows that are clearly and entirely missing.

**text_errors** — Text labels:
  - "Missing": a reference label is entirely absent → critical
  - "Garbled / unreadable": text is corrupted or illegible → critical
  - "Truncated": label is cut off or incomplete, showing only part
    of the expected word/phrase → critical
  - "Wrong text": label says something clearly different from reference → moderate
  - "Duplicated": same label appears where it shouldn't → moderate
  Do NOT report minor font, size, or positioning differences.

Severity guide:
  "critical" = fundamentally wrong or entirely missing
  "moderate" = clearly wrong but the element is present

Do NOT penalise components or text in the generated image that are missing
from the element list — they may be correct additions.

If the image is completely blank or corrupted, report aesthetic as "broken".

Respond ONLY with a JSON object:
{{
  "component_errors": [{{"d": "description", "s": "critical|moderate"}}],
  "arrow_errors": [{{"d": "description", "s": "critical|moderate"}}],
  "text_errors": [{{"d": "description", "s": "critical|moderate"}}],
  "aesthetic": "usable|broken"
}}"""


# ─── Dimension-specific prompts for independent 3-call mode ──────────

DIM_COMP_SYSTEM = (
    "You are a scientific diagram quality evaluator. You will see TWO images "
    "and a list of components extracted from the reference. "
    "LEFT = REFERENCE (ground truth). RIGHT = GENERATED (to evaluate). "
    "Your ONLY job is to check components/blocks/shapes. "
    "Do NOT report minor stylistic differences or speculative issues."
)

DIM_COMP_TEMPLATE = """## Context
You are evaluating a GENERATED scientific diagram. Focus ONLY on components.

The diagram should depict: "{prompt}"
LEFT image = REFERENCE. RIGHT image = GENERATED.

## Reference Components ({n_comp} total)
{comp_list}

## Your Task
Two checks:
1. **Reference check**: For each component listed above, check if it is present
   and correct in the GENERATED image.
2. **Hallucinated check**: Look at the GENERATED image and identify any major
   components/blocks that are clearly **wrong** — they do not correspond to
   ANY reference component, are obviously out-of-place, or look like random
   noise/artifacts that a diffusion model would produce.
   Only report components that are obviously erroneous; do NOT penalise minor
   decorative additions or stylistic differences.

Error types (use the FIRST matching type):
  1. "Missing": a reference component is entirely absent → critical
  2. "Wrong": a component is present at roughly the right location but depicts
     a clearly different concept from the reference → critical
  3. "Structural mismatch": the component exists at roughly the right location,
     but its internal structure differs significantly from the reference
     (e.g. reference shows stacked tiles but generated shows a single flat block;
     reference shows a grid with cells but generated shows a plain rectangle;
     reference shows sub-elements inside a component but generated merges
     them into a featureless blob; dimension annotations like "M×N" on a
     matrix are missing) → critical
  4. "Distorted": component is present but its shape is severely distorted,
     blurred, or illegible compared to the reference → critical
  5. "Hallucinated": a major element in the generated image that clearly should
     not exist — does not match any reference component, looks like random
     noise, or is an obvious artifact → critical
  6. "Duplicate": appears more times than in reference → moderate

Report ONLY the types listed above. Do NOT invent other categories.
Do NOT report minor style differences (color, shadow, rounded vs sharp corners).

IMPORTANT: Start each error description with the type name:
  "Missing: ...", "Wrong: ...", "Hallucinated: ...", etc.

If the image is completely blank or corrupted, report aesthetic as "broken".

Respond ONLY with a JSON object:
{{
  "component_errors": [{{"d": "description", "s": "critical|moderate"}}],
  "aesthetic": "usable|broken"
}}"""

DIM_ARROW_SYSTEM = (
    "You are a scientific diagram quality evaluator. You will see TWO images "
    "and a list of arrows/connections extracted from the reference. "
    "LEFT = REFERENCE (ground truth). RIGHT = GENERATED (to evaluate). "
    "Your job is to check arrows/connections for missing and hallucinated errors. "
    "Do NOT report minor stylistic differences or speculative issues."
)

DIM_ARROW_TEMPLATE = """## Context
You are evaluating a GENERATED scientific diagram. Focus ONLY on arrows/connections.

The diagram should depict: "{prompt}"
LEFT image = REFERENCE. RIGHT image = GENERATED.

## Reference Arrows / Connections ({n_arrow} total)
{arrow_list}

## Your Task
Two checks:
1. **Missing check**: For each reference connection above, check if it exists
   in the GENERATED image. Report those that are clearly **absent**.
2. **Hallucinated check**: Look at the GENERATED image and identify any arrows
   or connections that are clearly **wrong** — they do not correspond to ANY
   reference connection above and they connect components that should NOT be
   connected, point to empty space, or appear as random/meaningless lines.
   Only report arrows that are obviously erroneous; do NOT penalise arrows
   that could be reasonable connections even if not in the reference list.

Error types:
  - "Missing": a reference connection is entirely absent → critical
  - "Hallucinated": an arrow in the generated image that clearly should not
    exist — connects wrong components, points nowhere, or is random noise.
    Do NOT report arrows that look like plausible connections → critical

Arrow direction, endpoints, and routing cannot be reliably judged from
rasterized diagram images. Therefore:
Do NOT report reversed arrows, wrong endpoints, routing differences,
curvature, thickness, color, or any directional errors.

IMPORTANT: For every error, start the description with the error type:
  "Missing: ..." or "Hallucinated: ..."

If the image is completely blank or corrupted, report aesthetic as "broken".

Respond ONLY with a JSON object:
{{
  "arrow_errors": [{{"d": "description", "s": "critical"}}],
  "aesthetic": "usable|broken"
}}"""

DIM_TEXT_SYSTEM = (
    "You are a scientific diagram quality evaluator specialising in text "
    "readability. You will see TWO images and a list of text labels extracted "
    "from the reference. "
    "LEFT = REFERENCE (ground truth). RIGHT = GENERATED (to evaluate). "
    "Diffusion models frequently produce garbled, corrupted, or nonsensical "
    "text in generated images. Your job is to catch every such case. "
    "Do NOT report minor stylistic differences or speculative issues."
)

DIM_TEXT_TEMPLATE = """## Context
You are evaluating a GENERATED scientific diagram. Focus ONLY on text labels.
Diffusion models frequently struggle with text rendering — producing garbled,
corrupted, or nonsensical characters instead of readable text. Pay special
attention to whether each label is actually legible.

The diagram should depict: "{prompt}"
LEFT image = REFERENCE. RIGHT image = GENERATED.

## Reference Text Labels ({n_text} total)
{text_list}

## Your Task
For **every** text label listed above, locate it in the GENERATED image and
check: (1) is it present? (2) is it legible or garbled? (3) does it say the
right thing?
Report only errors you are **confident** about.

Error types (in priority order — use the FIRST matching type):
  1. "Missing": a reference label is entirely absent → critical
  2. "Garbled / unreadable": any character-level corruption — misspelled words,
     swapped/extra/missing characters, distorted glyphs, or text that does not
     form the correct word even if partially recognisable. If you can tell the
     text was *intended* to be a certain label but characters are wrong, that
     is garbled, NOT "wrong text".
     Examples: "filterin" instead of "filtering", "classifif" instead of
     "classify", "artie" instead of "article", scrambled characters → critical
  3. "Truncated": label is cut off or incomplete, showing only part
     of the expected word/phrase → critical
  4. "Wrong text": label is fully legible but says something **completely**
     different — a different word/concept, not a corrupted version of the
     reference label → moderate
  5. "Duplicated": same label appears where it shouldn't → moderate

IMPORTANT: If a label has ANY character-level corruption (misspelling,
distortion, scrambled characters), classify it as "Garbled / unreadable",
even if the intended meaning is still guessable.

Do NOT report minor font, size, or positioning differences.
Do NOT penalise text in the generated image that is not in the element
list — it may be a correct addition.

If the image is completely blank or corrupted, report aesthetic as "broken".

Respond ONLY with a JSON object:
{{
  "text_errors": [{{"d": "description", "s": "critical|moderate"}}],
  "aesthetic": "usable|broken"
}}"""


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def encode_image_to_base64(path: str, max_edge: int = 1024) -> str:
    from PIL import ImageFile
    img = None
    for attempt in range(2):
        _orig = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(path).convert("RGB")
            break
        except (OSError, SyntaxError):
            if attempt == 0:
                time.sleep(0.5)
            else:
                raise
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = _orig
    if img is None:
        raise OSError(f"Failed to load image: {path}")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def robust_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        depth, end = 0, -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    if start >= 0:
        fragment = text[start:]
        in_str = False
        for ci, ch in enumerate(fragment):
            if ch == '"' and (ci == 0 or fragment[ci - 1] != '\\'):
                in_str = not in_str
        if in_str:
            fragment += '"'
        for trim in range(0, min(len(fragment) // 2, 500)):
            candidate = fragment[:len(fragment) - trim] if trim else fragment
            open_b = candidate.count("{") - candidate.count("}")
            open_sq = candidate.count("[") - candidate.count("]")
            closers = "]" * max(0, open_sq) + "}" * max(0, open_b)
            try:
                return json.loads(candidate + closers)
            except json.JSONDecodeError:
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Scoring functions
# ═══════════════════════════════════════════════════════════════════════

_ERROR_TYPE_TO_SEVERITY: Dict[str, str] = {
    "missing": "critical", "wrong_identity": "critical", "shape_distortion": "critical",
    "duplication": "moderate", "duplicate": "moderate", "position_error": "moderate",
    "wrong_place": "moderate", "garbled": "critical", "wrong_content": "critical",
    "wrong": "minor", "wrong_target": "critical", "reversed": "critical",
    "spurious": "moderate", "no_arrowhead": "moderate", "style_mismatch": "minor",
}


def _classify_error_type(desc: str) -> str:
    """Extract error type from the description prefix."""
    d = desc.lower()
    if d.startswith("hallucinated") or "hallucinated" in d[:30]:
        return "hallucinated"
    if d.startswith("garbled") or "garbled" in d[:30] or "unreadable" in d[:30]:
        return "garbled"
    if d.startswith("missing") or "missing" in d[:15]:
        return "missing"
    if d.startswith("wrong") or "wrong" in d[:15]:
        return "wrong"
    if d.startswith("structural") or "structural" in d[:20]:
        return "structural"
    if d.startswith("distorted") or "distorted" in d[:20]:
        return "distorted"
    if d.startswith("duplicate") or "duplicate" in d[:20]:
        return "duplicate"
    if d.startswith("truncated") or "truncated" in d[:20]:
        return "truncated"
    return "other"


def _normalize_errors(error_list: List[Dict]) -> List[Dict]:
    _VALID_SEVS = {"critical", "moderate", "minor"}
    out = []
    for e in error_list:
        if not isinstance(e, dict):
            continue
        raw = e.get("severity") or e.get("s", "moderate")
        desc = e.get("d") or e.get("description", "")
        etype = _classify_error_type(desc)
        if raw in _VALID_SEVS:
            severity = raw
        else:
            key = raw.strip().lower().replace(" ", "_")
            severity = _ERROR_TYPE_TO_SEVERITY.get(key, "moderate")
        # V11e: text "wrong" → minor (readability bias correction)
        if etype == "wrong" and severity in ("critical", "moderate"):
            severity = "minor"
        out.append({"severity": severity, "etype": etype})
    return out


def compute_error_rate(errors: List[Dict], expected_count: int) -> float:
    if not errors:
        return 0.0
    weighted = sum(
        SEVERITY_WEIGHTS.get(e.get("severity", "moderate"), 0.5) for e in errors
    )
    denom = min(max(expected_count, MIN_EXPECTED), MAX_EXPECTED_CAP)
    return weighted / denom


def error_rate_to_score(rate: float, floor: float = 0.0) -> float:
    if rate <= 0.0:
        return 1.0
    return max(floor, 1.0 - rate)


def compute_score(abstraction: Dict[str, Any], error_result: Dict[str, Any]) -> float:
    n_comp = len(abstraction.get("components", []))
    n_arrow = len(abstraction.get("arrows", []))
    n_text = len(abstraction.get("text_labels", []))

    errors = error_result.get("errors", error_result) \
        if isinstance(error_result.get("errors"), dict) else error_result

    _DIM_INFO = [
        ("component", _normalize_errors(errors.get("component_errors", [])), n_comp),
        ("arrow", _normalize_errors(errors.get("arrow_errors", [])), n_arrow),
        ("text", _normalize_errors(errors.get("text_errors", [])), n_text),
    ]

    dim_weights = {"component": 1.0, "arrow": 1.0, "text": 1.0}
    weighted_parts: List[Tuple[float, float]] = []
    for dim_name, err_list, n_expected in _DIM_INFO:
        if n_expected > 0:
            w = dim_weights[dim_name]
            rate = compute_error_rate(err_list, n_expected)
            weighted_parts.append((w, rate))

    if not weighted_parts:
        total_errs = sum(len(el) for _, el, _ in _DIM_INFO)
        return 1.0 if total_errs == 0 else 0.5

    total_w = sum(w for w, _ in weighted_parts)
    score = sum(w * error_rate_to_score(rate) for w, rate in weighted_parts) / total_w
    return round(max(0.01, min(1.0, score)), 4)


def is_blank_image(error_result: Dict[str, Any]) -> bool:
    if not error_result:
        return False
    aesthetic = error_result.get("aesthetic", "usable")
    if isinstance(aesthetic, str) and aesthetic.strip().lower() != "broken":
        return False
    errors = error_result.get("errors", error_result) \
        if isinstance(error_result.get("errors"), dict) else error_result
    has_errors = any(len(errors.get(k, [])) for k in
                     ("component_errors", "arrow_errors", "text_errors"))
    return not has_errors


def format_abstraction_for_stage2(abstraction: Dict[str, Any], prompt: str) -> str:
    comps = abstraction.get("components", [])
    arrows = abstraction.get("arrows", [])
    texts = abstraction.get("text_labels", [])

    comp_lines = []
    for i, c in enumerate(comps, 1):
        name = c.get("name", "?")
        desc = c.get("description", "")
        comp_lines.append(f"  {i}. {name}" + (f" — {desc}" if desc else ""))
    comp_list = "\n".join(comp_lines) if comp_lines else "  (none detected)"

    arrow_lines = []
    for i, a in enumerate(arrows, 1):
        src = a.get("source", "?")
        tgt = a.get("target", "?")
        lbl = a.get("label") or ""
        arrow_lines.append(f"  {i}. {src} → {tgt}" + (f"  [{lbl}]" if lbl else ""))
    arrow_list = "\n".join(arrow_lines) if arrow_lines else "  (none detected)"

    text_lines = []
    for i, t in enumerate(texts, 1):
        txt = t.get("text", "?")
        loc = t.get("location", "?")
        text_lines.append(f'  {i}. "{txt}" at {loc}')
    text_list = "\n".join(text_lines) if text_lines else "  (none detected)"

    return STAGE2_USER_TEMPLATE.format(
        prompt=prompt[:300],
        n_comp=len(comps), n_arrow=len(arrows), n_text=len(texts),
        comp_list=comp_list, arrow_list=arrow_list, text_list=text_list,
    )


def _format_comp_list(abstraction: Dict[str, Any]) -> str:
    comps = abstraction.get("components", [])
    lines = []
    for i, c in enumerate(comps, 1):
        name = c.get("name", "?")
        desc = c.get("description", "")
        lines.append(f"  {i}. {name}" + (f" — {desc}" if desc else ""))
    return "\n".join(lines) if lines else "  (none detected)"


def _format_arrow_list(abstraction: Dict[str, Any]) -> str:
    arrows = abstraction.get("arrows", [])
    lines = []
    for i, a in enumerate(arrows, 1):
        src = a.get("source", "?")
        tgt = a.get("target", "?")
        lbl = a.get("label") or ""
        lines.append(f"  {i}. {src} → {tgt}" + (f"  [{lbl}]" if lbl else ""))
    return "\n".join(lines) if lines else "  (none detected)"


def _format_text_list(abstraction: Dict[str, Any]) -> str:
    texts = abstraction.get("text_labels", [])
    lines = []
    for i, t in enumerate(texts, 1):
        txt = t.get("text", "?")
        loc = t.get("location", "?")
        lines.append(f'  {i}. "{txt}" at {loc}')
    return "\n".join(lines) if lines else "  (none detected)"


def format_dim_prompt(dim: str, abstraction: Dict[str, Any], prompt: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a single dimension."""
    p = prompt[:300]
    if dim == "component":
        user = DIM_COMP_TEMPLATE.format(
            prompt=p,
            n_comp=len(abstraction.get("components", [])),
            comp_list=_format_comp_list(abstraction),
        )
        return DIM_COMP_SYSTEM, user
    elif dim == "arrow":
        user = DIM_ARROW_TEMPLATE.format(
            prompt=p,
            n_arrow=len(abstraction.get("arrows", [])),
            arrow_list=_format_arrow_list(abstraction),
        )
        return DIM_ARROW_SYSTEM, user
    else:  # text
        user = DIM_TEXT_TEMPLATE.format(
            prompt=p,
            n_text=len(abstraction.get("text_labels", [])),
            text_list=_format_text_list(abstraction),
        )
        return DIM_TEXT_SYSTEM, user


def score_single(
    abstraction: Dict[str, Any],
    s2_result: Optional[Dict[str, Any]],
    default_score: float = 0.1,
) -> Dict[str, Any]:
    if s2_result is None:
        return {
            "score": default_score, "status": "parse_failed",
            "component_score": None, "arrow_score": None, "text_score": None,
            "component_errors": 0, "arrow_errors": 0, "text_errors": 0,
        }
    if is_blank_image(s2_result):
        return {
            "score": 0.0, "status": "blank",
            "component_score": None, "arrow_score": None, "text_score": None,
            "component_errors": 0, "arrow_errors": 0, "text_errors": 0,
        }

    errors = s2_result.get("errors", s2_result) \
        if isinstance(s2_result.get("errors"), dict) else s2_result

    n_comp = len(abstraction.get("components", []))
    n_arrow = len(abstraction.get("arrows", []))
    n_text = len(abstraction.get("text_labels", []))

    ce = _normalize_errors(errors.get("component_errors", []))
    ae = _normalize_errors(errors.get("arrow_errors", []))
    te = _normalize_errors(errors.get("text_errors", []))

    comp_rate = compute_error_rate(ce, n_comp) if n_comp > 0 else None
    arrow_rate = compute_error_rate(ae, n_arrow) if n_arrow > 0 else None
    text_rate = compute_error_rate(te, n_text) if n_text > 0 else None

    comp_score = round(error_rate_to_score(comp_rate), 4) if comp_rate is not None else None
    arrow_score = round(error_rate_to_score(arrow_rate), 4) if arrow_rate is not None else None
    text_score = round(error_rate_to_score(text_rate), 4) if text_rate is not None else None

    overall = compute_score(abstraction, s2_result)

    return {
        "score": overall, "status": "ok",
        "component_score": comp_score, "arrow_score": arrow_score, "text_score": text_score,
        "component_errors": len(errors.get("component_errors", [])),
        "arrow_errors": len(errors.get("arrow_errors", [])),
        "text_errors": len(errors.get("text_errors", [])),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Round-robin Azure OpenAI client pool
# ═══════════════════════════════════════════════════════════════════════

class ClientPool:
    """Thread-safe round-robin pool of OpenAI-compatible clients.

    Supports three backends configured via eval/judge_config.py:
      - "openai"        Standard OpenAI API (api key)
      - "azure_apikey"  Azure OpenAI with per-endpoint API keys
      - "azure_cli"     Azure OpenAI with AzureCliCredential (az login)
    """

    def __init__(self, deployment_name: str, auth_mode: str = "cli",
                 endpoints: Optional[Dict[str, str]] = None):
        from openai import OpenAI

        self._lock = threading.Lock()
        self._clients: list[tuple] = []
        self._blacklisted: set = set()

        # ── openai backend: single client, standard API key ──────────────
        if _JUDGE_BACKEND == "openai":
            try:
                import judge_config as _jc
                api_key  = getattr(_jc, "OPENAI_API_KEY",  "")
                base_url = getattr(_jc, "OPENAI_BASE_URL", "https://api.openai.com/v1")
                model    = getattr(_jc, "OPENAI_MODEL",    deployment_name)
            except ImportError:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                base_url = "https://api.openai.com/v1"
                model = deployment_name
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Add it to eval/judge_config.py "
                    "or set the OPENAI_API_KEY environment variable."
                )
            client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
            self._clients.append((client, model, base_url))
            logger.info(f"Client pool ready: 1 OpenAI endpoint (model={model})")
            self._idx = 0
            return

        # ── azure_apikey backend: {endpoint: api_key} ────────────────────
        if _JUDGE_BACKEND == "azure_apikey":
            eps = endpoints or ENDPOINT_TOKEN_SCOPE
            logger.info(f"Initialising client pool ({len(eps)} Azure endpoints, auth=apikey) ...")
            _failed = 0
            for endpoint, api_key in eps.items():
                try:
                    client = OpenAI(base_url=endpoint, api_key=api_key, max_retries=0)
                    self._clients.append((client, deployment_name, endpoint))
                except Exception:
                    _failed += 1
            if _failed:
                logger.warning(f"  {_failed} endpoint(s) failed to initialise")
            if not self._clients:
                raise RuntimeError("No Azure API-key clients could be initialised.")
            logger.info(f"Client pool ready: {len(self._clients)} endpoints")
            self._idx = 0
            return

        # ── azure_cli backend (default): AzureCliCredential ──────────────
        eps = endpoints or ENDPOINT_TOKEN_SCOPE
        logger.info(f"Initialising client pool ({len(eps)} endpoints, auth={auth_mode}) ...")
        _failed = 0
        for endpoint, scope in eps.items():
            try:
                token_fn = self._get_token_fn(scope, auth_mode)
                client = OpenAI(base_url=endpoint, api_key=token_fn, max_retries=0)
                self._clients.append((client, deployment_name, endpoint))
            except Exception:
                _failed += 1
        if _failed:
            logger.warning(f"  {_failed} endpoint(s) failed to initialise")

        if not self._clients:
            raise RuntimeError(
                "No Azure OpenAI clients could be initialised. "
                "Configure eval/judge_config.py (see judge_config.example.py)."
            )
        logger.info(f"Client pool ready: {len(self._clients)} endpoints")
        self._idx = 0

    @staticmethod
    def _get_token_fn(scope: str, auth_mode: str):
        from azure.identity import (
            AzureCliCredential, DefaultAzureCredential,
            ManagedIdentityCredential, get_bearer_token_provider,
        )
        if auth_mode == "managed_identity" and AZURE_CLIENT_ID:
            credential = ManagedIdentityCredential(client_id=AZURE_CLIENT_ID)
        elif auth_mode == "cli":
            credential = AzureCliCredential()
        else:
            credential = DefaultAzureCredential()
        return get_bearer_token_provider(credential, scope)

    def blacklist(self, endpoint: str):
        """Mark an endpoint as permanently unavailable for this deployment."""
        with self._lock:
            if endpoint not in self._blacklisted:
                self._blacklisted.add(endpoint)
                logger.warning(f"  Blacklisted endpoint ({len(self._blacklisted)}/{len(self._clients)}): {endpoint}")

    def next(self) -> tuple:
        """Return (client, deployment_name, endpoint_url), skipping blacklisted."""
        with self._lock:
            n = len(self._clients)
            for _ in range(n):
                entry = self._clients[self._idx % n]
                self._idx += 1
                if entry[2] not in self._blacklisted:
                    return entry
            # All blacklisted — fall back to any
            entry = self._clients[self._idx % n]
            self._idx += 1
            return entry

    @property
    def size(self) -> int:
        return len(self._clients) - len(self._blacklisted)

    @property
    def total(self) -> int:
        return len(self._clients)


# ═══════════════════════════════════════════════════════════════════════
#  Data loading — benchmark_final/
# ═══════════════════════════════════════════════════════════════════════

def _detect_gen_images(gen_level_dir: Path) -> Optional[callable]:
    """
    Auto-detect how generated images are named inside gen_level_dir.
    Returns a callable: idx -> Optional[str] (absolute path string), or None.

    Uses os.listdir + dict for NFS performance (avoids per-file stat calls).
    """
    gen_level_str = str(gen_level_dir)

    # Auto-detect subdir (e.g. cfg_3.5, cfg_7.0), ignore prompt_ dirs and hidden
    try:
        top_entries = os.listdir(gen_level_str)
    except OSError:
        return None
    cfg_subdirs = [
        e for e in top_entries
        if not e.startswith(".") and not e.startswith("prompt")
        and os.path.isdir(os.path.join(gen_level_str, e))
    ]
    if cfg_subdirs:
        gen_img_dir = os.path.join(gen_level_str, sorted(cfg_subdirs)[0])
        logger.info(f"    Using subdir: {os.path.basename(gen_img_dir)}")
    else:
        gen_img_dir = gen_level_str

    # List all files once
    try:
        all_names = os.listdir(gen_img_dir)
    except OSError:
        return None
    all_names_set = set(all_names)

    # (A) Numeric: 5-digit / 4-digit
    for fmt, test_name in [
        ("{:05d}.png", "00000.png"), ("{:05d}.jpg", "00000.jpg"),
        ("{:04d}.png", "0000.png"), ("{:04d}.jpg", "0000.jpg"),
    ]:
        if test_name in all_names_set:
            _dir = gen_img_dir
            _fmt = fmt
            _names = all_names_set
            logger.info(f"    Pattern: numeric ({test_name}) in {os.path.basename(_dir)}")
            def _gen_a(i, _d=_dir, _f=_fmt, _ns=_names):
                name = _f.format(i)
                return os.path.join(_d, name) if name in _ns else None
            return _gen_a

    # (B) promptNNNN_*.png  (e.g. prompt0000_The_figure.png)
    prompt_files = [n for n in all_names if n.startswith("prompt") and "_" in n
                    and n.lower().rsplit(".", 1)[-1] in ("png", "jpg", "jpeg")]
    if prompt_files:
        lookup: Dict[int, str] = {}
        for n in prompt_files:
            stem = n.rsplit(".", 1)[0]
            idx_str = stem.split("_")[0].replace("prompt", "")
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            lookup[idx] = os.path.join(gen_img_dir, n)
        if lookup:
            logger.info(f"    Pattern: promptNNNN_*.png ({len(lookup)} files)")
            return lambda i, _lk=lookup: _lk.get(i)

    # (C) prompt_NNNN/ directory structure
    prompt_dirs = [n for n in all_names if n.startswith("prompt_")
                   and os.path.isdir(os.path.join(gen_img_dir, n))]
    if not prompt_dirs:
        # Also check top-level if gen_img_dir is a subdir
        if gen_img_dir != gen_level_str:
            prompt_dirs = [n for n in top_entries if n.startswith("prompt_")
                          and os.path.isdir(os.path.join(gen_level_str, n))]
            if prompt_dirs:
                gen_img_dir = gen_level_str
    if prompt_dirs:
        dir_lookup: Dict[int, str] = {}
        for dn in prompt_dirs:
            idx_str = dn.replace("prompt_", "")
            try:
                pidx = int(idx_str)
            except ValueError:
                continue
            pd = os.path.join(gen_img_dir, dn)
            imgs = [f for f in os.listdir(pd) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if imgs:
                dir_lookup[pidx] = os.path.join(pd, sorted(imgs)[0])
        if dir_lookup:
            logger.info(f"    Pattern: prompt_NNNN/sample.png ({len(dir_lookup)} dirs)")
            return lambda i, _lk=dir_lookup: _lk.get(i)

    # (D) Fallback: any image, extract first number from filename
    img_names = [n for n in all_names if n.lower().rsplit(".", 1)[-1] in ("png", "jpg", "jpeg")]
    if img_names:
        idx_re = re.compile(r'(\d+)')
        fallback_lookup: Dict[int, str] = {}
        for n in img_names:
            nums = idx_re.findall(n.rsplit(".", 1)[0])
            if nums:
                fallback_lookup[int(nums[0])] = os.path.join(gen_img_dir, n)
        if fallback_lookup:
            sample = sorted(img_names)[0]
            logger.info(f"    Pattern: fallback numeric ({len(fallback_lookup)} files, e.g. {sample})")
            return lambda i, _lk=fallback_lookup: _lk.get(i)

    return None





def load_benchmark_entries_from_hf(
    hf_repo: str,
    gen_dir: str,
    levels: List[str],
    new_gen: bool = False,
    hf_token: str = None,
    gt_cache_dir: str = None,
) -> List[Dict[str, Any]]:
    """
    Load ALL benchmark data from HuggingFace microsoft/SciFormaBench.
    GT images + prompts + rubrics all come from HF — no local files needed.
    """
    from datasets import load_dataset
    import tempfile

    token = hf_token or os.environ.get("HF_TOKEN")
    if gt_cache_dir is None:
        gt_cache_dir = tempfile.mkdtemp(prefix="sciformabench_gt_")
    gt_cache = Path(gt_cache_dir)

    entries = []
    for level in levels:
        logger.info(f"[{level}] Downloading from {hf_repo} ...")
        try:
            ds = load_dataset(hf_repo, level, split="test", token=token)
        except Exception as e:
            logger.warning(f"  Could not load {level} from HF: {e}")
            continue
        logger.info(f"  {level}: {len(ds)} rows")

        gen_level_dir = None
        gen_dir_path = Path(gen_dir)
        for alias in [level, "easy"] + [f"{level}_{n}" for n in [500, 900, 600, 800, 1000]]:
            for cfg in ["cfg_4.0", "cfg_3.5", ""]:
                c = gen_dir_path / alias / cfg if cfg else gen_dir_path / alias
                if c.exists() and any(c.iterdir()):
                    gen_level_dir = c
                    break
            if gen_level_dir:
                break

        if gen_level_dir is None:
            logger.warning(f"  [{level}] No generated images found under {gen_dir}")
            continue

        gen_lookup = _detect_gen_images(Path(gen_level_dir))
        if gen_lookup is None:
            logger.warning(f"  [{level}] Could not detect gen image naming")
            continue

        skipped = 0
        for local_idx, row in enumerate(ds):
            gen_path = gen_lookup(local_idx)
            if not gen_path or not os.path.exists(gen_path):
                skipped += 1
                continue

            gt_dest = gt_cache / row["year"] / row["image_path"]
            if not gt_dest.exists():
                gt_dest.parent.mkdir(parents=True, exist_ok=True)
                row["gt_image"].save(str(gt_dest))

            abst = {
                "components":    json.loads(row.get("rubric_components", "[]")),
                "arrows":        json.loads(row.get("rubric_arrows", "[]")),
                "text_elements": json.loads(row.get("rubric_text", "[]")),
            }
            entries.append({
                "level":       level,
                "local_idx":   local_idx,
                "global_idx":  local_idx,
                "prompt":      row["prompt"],
                "gt_path":     str(gt_dest),
                "gen_path":    gen_path,
                "image_path":  row.get("image_path", ""),
                "year":        row.get("year", ""),
                "abstraction": abst,
            })

        logger.info(f"  [{level}] loaded={len([e for e in entries if e['level']==level])}, skipped={skipped}")

    return entries


def load_benchmark_entries(
    benchmark_dir: str,
    gt_base: str,
    gen_dir: str,
    levels: List[str],
    internal: bool = False,
    new_gen: bool = False,
    use_mask: bool = False,
    mask_file: Optional[str] = None,
    rubrics_suffix: str = "",
) -> List[Dict[str, Any]]:
    """
    Load benchmark entries.

    Clean mode (internal=False):
      benchmark_simple.json  →  {validation_prompts, gt_images[{year, image_path}]}
      benchmark_simple_rubrics.json  →  {entries[{index, image_path, stage1_abstraction}]}
      Abstraction lookup by array index.

    Internal mode (internal=True):
      benchmark_final_from2k4/benchmark_simple.json  →  {validation_prompts, metadata[{globalIdx, year, image_path, ...}]}
      benchmark_final_from2k4/benchmark_simple_rubrics.json  →  {entries[{global_idx, image_path, stage1_abstraction}]}
      Abstraction lookup by globalIdx.
    """
    entries = []
    bench_dir = Path(benchmark_dir)
    if internal:
        bench_dir = bench_dir / "benchmark_final_from2k4"

    # Load mask if needed
    mask_indices: Optional[Dict[str, List[int]]] = None
    if use_mask and mask_file:
        with open(mask_file, encoding="utf-8") as f:
            mask_data = json.load(f)
        mask_indices = mask_data.get("indices", {})
        logger.info(f"Using mask from {mask_file}")

    for level in levels:
        # Support two directory layouts:
        #   1. New layout (SciForma repo):  prompts/{level}.json + rubrics/{level}.json
        #      where level names are: easy / medium / hard
        #   2. Legacy layout (benchmark_final): benchmark_{level}.json + benchmark_{level}_rubrics.json
        #      where level names are: simple / medium / hard

        # Level name alias: "easy" == "simple"
        bench_name  = "simple" if level == "easy" else level
        display_lvl = level   # keep original name for output keys

        # Try new layout first, fall back to legacy
        new_bench  = bench_dir / "prompts" / f"{level}.json"
        new_rubric = bench_dir / "rubrics"  / f"{level}.json"
        old_bench  = bench_dir / f"benchmark_{bench_name}.json"
        old_rubric = bench_dir / f"benchmark_{bench_name}_rubrics{rubrics_suffix}.json"

        if new_bench.exists() and new_rubric.exists():
            bench_file  = new_bench
            rubric_file = new_rubric
        elif old_bench.exists() and old_rubric.exists():
            bench_file  = old_bench
            rubric_file = old_rubric
        else:
            logger.warning(f"Files not found for level '{level}' in {bench_dir}")
            continue

        with open(bench_file, encoding="utf-8") as f:
            bench_data = json.load(f)
        with open(rubric_file, encoding="utf-8") as f:
            rubric_data = json.load(f)

        prompts = bench_data["validation_prompts"]

        # Build abstraction lookup
        abst_lookup: Dict[int, Dict] = {}
        for entry in rubric_data.get("entries", []):
            abst = entry.get("stage1_abstraction")
            if not abst:
                continue
            if internal:
                # Internal: key by global_idx
                gidx = entry.get("global_idx")
                if gidx is not None:
                    abst_lookup[gidx] = abst
            else:
                # Clean: key by index (0-based array position)
                idx = entry.get("index")
                if idx is not None:
                    abst_lookup[idx] = abst

        # Build gt_images info (year, image_path, old_local_idx) per local index
        if internal:
            metadata_list = bench_data["metadata"]
            gt_info = [
                {"year": m.get("year", "2024"), "image_path": m.get("image_path", ""),
                 "globalIdx": m.get("globalIdx"),
                 "old_local_idx": m.get("old_local_idx")}
                for m in metadata_list
            ]
        else:
            gt_images_list = bench_data["gt_images"]
            gt_info = [
                {"year": g.get("year", "2024"), "image_path": g.get("image_path", ""),
                 "globalIdx": None,
                 "old_local_idx": g.get("old_local_idx")}
                for g in gt_images_list
            ]

        # Determine which local indices to evaluate
        if mask_indices and level in mask_indices:
            local_indices = sorted(mask_indices[level])
        else:
            local_indices = list(range(len(prompts)))

        # Detect gen image naming — try multiple level dir aliases
        # Covers: simple, simple_500, simple_600, easy, etc.
        level_aliases = [level, f"{level}_{len(prompts)}"]
        if level == "simple":
            level_aliases.append("easy")
        # Also try old benchmark sizes (600/1000/800) for backward compat
        _OLD_SIZES = {"simple": [600], "medium": [1000], "hard": [800]}
        for old_n in _OLD_SIZES.get(level, []):
            alias = f"{level}_{old_n}"
            if alias not in level_aliases:
                level_aliases.append(alias)
        gen_level_dir = None
        gen_dir_path = Path(gen_dir)
        for alias in level_aliases:
            candidate = gen_dir_path / alias
            if candidate.exists():
                gen_level_dir = candidate
                break
        # Glob fallback: match {level}_* dirs
        if gen_level_dir is None and gen_dir_path.exists():
            for d in sorted(gen_dir_path.iterdir()):
                if d.is_dir() and d.name.startswith(level + "_"):
                    gen_level_dir = d
                    break
        if gen_level_dir is None:
            logger.warning(f"Gen dir not found for {level}, tried: {level_aliases}")
            continue

        logger.info(f"  [{level}] gen_dir={gen_level_dir}")
        gen_fmt = _detect_gen_images(gen_level_dir)
        if gen_fmt is None:
            try:
                files = os.listdir(str(gen_level_dir))[:10]
                logger.warning(f"  [{level}] No gen images detected. Files: {files}")
            except Exception:
                logger.warning(f"  [{level}] No gen images in {gen_level_dir}")
            continue

        skipped_no_abst = 0
        skipped_no_gen = 0
        for local_idx in local_indices:
            if local_idx >= len(prompts):
                continue
            info = gt_info[local_idx]
            year = info["year"]
            image_path = info["image_path"]
            global_idx = info["globalIdx"]
            old_local_idx = info.get("old_local_idx")

            gt_path = os.path.join(gt_base, str(year), image_path)

            # For gen images:
            #   new_gen=True  → images numbered sequentially 0..N-1 (colleagues' 2k inference)
            #   new_gen=False → images numbered by old benchmark index (old 600/1000/800 runs)
            if new_gen:
                gen_lookup_idx = local_idx
            else:
                gen_lookup_idx = old_local_idx if old_local_idx is not None else local_idx
            gen_path = gen_fmt(gen_lookup_idx)  # returns str or None

            # Lookup abstraction
            if internal:
                abstraction = abst_lookup.get(global_idx)
            else:
                abstraction = abst_lookup.get(local_idx)

            if not abstraction:
                skipped_no_abst += 1
                continue
            if not gen_path:
                skipped_no_gen += 1
                continue

            entries.append({
                "level": level,
                "local_idx": local_idx,
                "global_idx": global_idx if global_idx is not None else local_idx,
                "prompt": prompts[local_idx],
                "gt_path": gt_path,
                "gen_path": gen_path,
                "abstraction": abstraction,
                "year": year,
                "image_path": image_path,
            })

        if skipped_no_abst:
            logger.warning(f"  [{level}] Skipped {skipped_no_abst} entries (no abstraction)")
        if skipped_no_gen:
            logger.warning(f"  [{level}] Skipped {skipped_no_gen} entries (gen image missing)")

    return entries


# ═══════════════════════════════════════════════════════════════════════
#  GPT-5.4 VLM inference with round-robin
# ═══════════════════════════════════════════════════════════════════════

def _call_gpt54_single(
    client,
    deployment_name: str,
    gt_path: str,
    gen_path: str,
    abstraction: Dict[str, Any],
    prompt: str,
    max_edge: int = 1024,
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
) -> tuple[Optional[Dict], str]:
    gt_b64 = encode_image_to_base64(gt_path, max_edge=max_edge)
    gen_b64 = encode_image_to_base64(gen_path, max_edge=max_edge)
    user_text = format_abstraction_for_stage2(abstraction, prompt)

    messages = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    api_kwargs: Dict[str, Any] = dict(
        model=deployment_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    if reasoning_effort:
        api_kwargs["reasoning_effort"] = reasoning_effort
    else:
        api_kwargs["temperature"] = temperature
        api_kwargs["seed"] = 42

    resp = client.chat.completions.create(
        **api_kwargs,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "diagram_eval",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "component_errors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "d": {"type": "string"},
                                    "s": {"type": "string", "enum": ["critical", "moderate"]}
                                },
                                "required": ["d", "s"],
                                "additionalProperties": False
                            }
                        },
                        "arrow_errors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "d": {"type": "string"},
                                    "s": {"type": "string", "enum": ["critical", "moderate"]}
                                },
                                "required": ["d", "s"],
                                "additionalProperties": False
                            }
                        },
                        "text_errors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "d": {"type": "string"},
                                    "s": {"type": "string", "enum": ["critical", "moderate"]}
                                },
                                "required": ["d", "s"],
                                "additionalProperties": False
                            }
                        },
                        "aesthetic": {"type": "string", "enum": ["usable", "broken"]}
                    },
                    "required": ["component_errors", "arrow_errors", "text_errors", "aesthetic"],
                    "additionalProperties": False
                }
            }
        },
    )
    raw = resp.choices[0].message.content if resp.choices else ""
    parsed = robust_json_parse(raw)
    return parsed, raw


# JSON schemas for individual dimension calls
_DIM_SCHEMA = {
    "component": {
        "type": "json_schema",
        "json_schema": {
            "name": "comp_eval", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "component_errors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "d": {"type": "string"},
                                "s": {"type": "string", "enum": ["critical", "moderate"]}
                            },
                            "required": ["d", "s"], "additionalProperties": False
                        }
                    },
                    "aesthetic": {"type": "string", "enum": ["usable", "broken"]}
                },
                "required": ["component_errors", "aesthetic"],
                "additionalProperties": False
            }
        }
    },
    "arrow": {
        "type": "json_schema",
        "json_schema": {
            "name": "arrow_eval", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "arrow_errors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "d": {"type": "string"},
                                "s": {"type": "string", "enum": ["critical", "moderate"]}
                            },
                            "required": ["d", "s"], "additionalProperties": False
                        }
                    },
                    "aesthetic": {"type": "string", "enum": ["usable", "broken"]}
                },
                "required": ["arrow_errors", "aesthetic"],
                "additionalProperties": False
            }
        }
    },
    "text": {
        "type": "json_schema",
        "json_schema": {
            "name": "text_eval", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "text_errors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "d": {"type": "string"},
                                "s": {"type": "string", "enum": ["critical", "moderate"]}
                            },
                            "required": ["d", "s"], "additionalProperties": False
                        }
                    },
                    "aesthetic": {"type": "string", "enum": ["usable", "broken"]}
                },
                "required": ["text_errors", "aesthetic"],
                "additionalProperties": False
            }
        }
    },
}


def _call_gpt54_dim(
    client,
    deployment_name: str,
    gt_b64: str,
    gen_b64: str,
    dim: str,
    abstraction: Dict[str, Any],
    prompt: str,
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
) -> tuple[Optional[Dict], str]:
    """Call GPT for a single dimension (component/arrow/text)."""
    system_text, user_text = format_dim_prompt(dim, abstraction, prompt)

    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    api_kwargs: Dict[str, Any] = dict(
        model=deployment_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    if reasoning_effort:
        api_kwargs["reasoning_effort"] = reasoning_effort
    else:
        api_kwargs["temperature"] = temperature
        api_kwargs["seed"] = 42

    resp = client.chat.completions.create(
        **api_kwargs,
        response_format=_DIM_SCHEMA[dim],
    )
    raw = resp.choices[0].message.content if resp.choices else ""
    parsed = robust_json_parse(raw)
    return parsed, raw


def _call_dim_with_retry(
    dim: str,
    pool: ClientPool,
    gt_b64: str,
    gen_b64: str,
    abstraction: Dict[str, Any],
    prompt: str,
    entry_label: str,
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
) -> tuple[Optional[Dict], str]:
    """Call a single dimension with retry logic."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        client, deployment, endpoint = pool.next()
        time.sleep(random.uniform(0.01, 0.5))
        try:
            parsed, raw = _call_gpt54_dim(
                client, deployment, gt_b64, gen_b64,
                dim, abstraction, prompt,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            return parsed, raw
        except Exception as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            err_msg = str(exc)
            if (status in (400, 404)
                    and ("DeploymentNotFound" in err_msg or "unknown_model" in err_msg)):
                pool.blacklist(endpoint)
                continue
            if status is not None and status not in RETRY_CODES:
                logger.warning(f"  [{entry_label}/{dim}] HTTP {status} (no retry): {exc}")
                return None, str(exc)
            wait = 3 * (2 ** attempt) + random.uniform(0, 2)
            logger.warning(f"  [{entry_label}/{dim}] attempt {attempt+1}/{MAX_RETRIES}, "
                         f"retry in {wait:.0f}s: {exc}")
            time.sleep(wait)

    logger.error(f"  [{entry_label}/{dim}] FAILED after {MAX_RETRIES} attempts: {last_exc}")
    return None, str(last_exc) if last_exc else ""


def eval_one_entry(
    entry: Dict[str, Any],
    pool: ClientPool,
    max_edge: int = 1024,
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    split_dims: bool = False,
) -> tuple[Optional[Dict], str]:
    if not split_dims:
        # Original single-call mode
        last_exc = None
        for attempt in range(MAX_RETRIES):
            client, deployment, endpoint = pool.next()
            time.sleep(random.uniform(0.01, 1.0))
            try:
                parsed, raw = _call_gpt54_single(
                    client, deployment,
                    entry["gt_path"], entry["gen_path"],
                    entry["abstraction"], entry["prompt"],
                    max_edge=max_edge,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
                return parsed, raw
            except Exception as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                err_msg = str(exc)
                if (status in (400, 404)
                        and ("DeploymentNotFound" in err_msg or "unknown_model" in err_msg)):
                    pool.blacklist(endpoint)
                    continue
                if status is not None and status not in RETRY_CODES:
                    logger.warning(f"  [{entry['level']}#{entry['local_idx']:05d}] "
                                 f"HTTP {status} (no retry): {exc}")
                    return None, str(exc)
                wait = 3 * (2 ** attempt) + random.uniform(0, 2)
                logger.warning(f"  [{entry['level']}#{entry['local_idx']:05d}] "
                             f"attempt {attempt+1}/{MAX_RETRIES}, retry in {wait:.0f}s: {exc}")
                time.sleep(wait)
        logger.error(f"  [{entry['level']}#{entry['local_idx']:05d}] "
                     f"FAILED after {MAX_RETRIES} attempts: {last_exc}")
        return None, str(last_exc) if last_exc else ""

    # ── Split-dims mode: 3 independent GPT calls ──
    try:
        gt_b64 = encode_image_to_base64(entry["gt_path"], max_edge=max_edge)
        gen_b64 = encode_image_to_base64(entry["gen_path"], max_edge=max_edge)
    except Exception as exc:
        entry_label = f"{entry['level']}#{entry['local_idx']:05d}"
        logger.warning(f"  [{entry_label}] image load failed, skipping: {exc}")
        return None, str(exc)
    entry_label = f"{entry['level']}#{entry['local_idx']:05d}"

    dim_results: Dict[str, tuple] = {}
    # Run 3 dims in parallel using threads
    with ThreadPoolExecutor(max_workers=3) as dim_executor:
        dim_futures = {}
        for dim in ("component", "arrow", "text"):
            fut = dim_executor.submit(
                _call_dim_with_retry,
                dim, pool, gt_b64, gen_b64,
                entry["abstraction"], entry["prompt"], entry_label,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            dim_futures[dim] = fut
        for dim, fut in dim_futures.items():
            dim_results[dim] = fut.result()

    # Merge into combined result
    merged: Dict[str, Any] = {
        "component_errors": [],
        "arrow_errors": [],
        "text_errors": [],
        "aesthetic": "usable",
    }
    raw_parts = {}
    for dim in ("component", "arrow", "text"):
        parsed, raw = dim_results[dim]
        raw_parts[dim] = raw
        if parsed is None:
            continue
        err_key = f"{dim}_errors"
        merged[err_key] = parsed.get(err_key, [])
        if parsed.get("aesthetic") == "broken":
            merged["aesthetic"] = "broken"

    combined_raw = json.dumps(raw_parts, ensure_ascii=False)
    return merged, combined_raw


def run_evaluation(
    entries: List[Dict[str, Any]],
    pool: ClientPool,
    workers: int = 16,
    max_edge: int = 1024,
    num_retest: int = 1,
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    split_dims: bool = False,
) -> List[Dict[str, Any]]:
    all_round_results: List[List[Optional[tuple]]] = []

    for round_idx in range(num_retest):
        logger.info(f"\n{'='*60}")
        logger.info(f"  ROUND {round_idx + 1} / {num_retest}")
        logger.info(f"{'='*60}")

        results: List[Optional[tuple]] = [None] * len(entries)
        done_count = 0

        def _worker(idx_entry):
            idx, entry = idx_entry
            parsed, raw = eval_one_entry(
                entry, pool, max_edge=max_edge,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                split_dims=split_dims,
            )
            return idx, (parsed, raw)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, (i, e)): i
                for i, e in enumerate(entries)
            }
            for fut in as_completed(futures):
                idx, result = fut.result()
                results[idx] = result
                done_count += 1
                if done_count % 100 == 0 or done_count == len(entries):
                    logger.info(f"  [{done_count}/{len(entries)}] evaluated")

        all_round_results.append(results)

    # Score & aggregate
    logger.info("\nScoring and aggregating ...")
    per_sample_summaries = []

    for i, entry in enumerate(entries):
        round_scores = []
        round_details = []
        for round_idx in range(num_retest):
            result_tuple = all_round_results[round_idx][i]
            if result_tuple is not None:
                s2_result, raw_text = result_tuple
            else:
                s2_result, raw_text = None, ""
            detail = score_single(entry["abstraction"], s2_result)
            detail["raw_output"] = raw_text
            detail["parsed_json"] = s2_result
            round_scores.append(detail["score"])
            round_details.append(detail)

        avg_score = float(np.mean(round_scores))
        std_score = float(np.std(round_scores)) if len(round_scores) > 1 else 0.0

        avg_aspects = {}
        for aspect in ("component_score", "arrow_score", "text_score"):
            vals = [d[aspect] for d in round_details if d[aspect] is not None]
            avg_aspects[aspect] = round(float(np.mean(vals)), 4) if vals else None

        per_sample_summaries.append({
            "level": entry["level"],
            "local_idx": entry["local_idx"],
            "global_idx": entry["global_idx"],
            "prompt": entry["prompt"],
            "gt_path": entry["gt_path"],
            "gen_path": entry["gen_path"],
            "image_path": entry.get("image_path", ""),
            "avg_score": round(avg_score, 4),
            "std_score": round(std_score, 4),
            "num_rounds": num_retest,
            "round_scores": [round(s, 4) for s in round_scores],
            **avg_aspects,
            "round_details": round_details,
        })

    return per_sample_summaries


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark V9r rubrics evaluation with GPT-5.4 (round-robin Azure endpoints)"
    )
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Root dir of generated images (expects <gen_dir>/<level>/ subfolders)")
    parser.add_argument("--gt_base", type=str,
                        default=os.environ.get("SCIFORMA_GT_BASE", ""),
                        help="Local root dir of GT images (year/paper_id/img.png). "
                             "If not set, GT images are downloaded from --hf_benchmark automatically.")
    parser.add_argument("--benchmark_dir", type=str,
                        default=str(Path(__file__).parent),
                        help="Dir containing prompts/ + rubrics/ (default: this script's dir)")
    parser.add_argument("--hf_benchmark", type=str,
                        default="microsoft/SciFormaBench",
                        help="HuggingFace dataset ID for GT images (default: microsoft/SciFormaBench). "
                             "Used automatically when --gt_base is not set. Set to '' to disable.")
    parser.add_argument("--internal", action="store_true",
                        help="Use internal format (benchmark_final_from2k4/) with globalIdx metadata")
    parser.add_argument("--new_gen", action="store_true", default=True,
                        help="(Default: True) Generated images are numbered sequentially 0..N-1. "
                             "Set --no_new_gen only if using old-format benchmark images.")
    parser.add_argument("--no_new_gen", dest="new_gen", action="store_false")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Where to write per-sample and summary JSONs")
    parser.add_argument("--levels", nargs="+",
                        default=["simple", "medium", "hard"],
                        choices=["simple", "medium", "hard"])
    parser.add_argument("--num_retest", type=int, default=1,
                        help="Number of eval rounds per sample (averaged)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Concurrent API request workers (default: 4 * num_endpoints)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for GPT API calls (0 = deterministic)")
    parser.add_argument("--max_completion_tokens", type=int, default=16384,
                        help="Max completion tokens for GPT API")
    parser.add_argument("--deployment_name", type=str, default="gpt-5.4",
                        help="Azure OpenAI deployment name")
    parser.add_argument("--auth", type=str, default="cli",
                        choices=["cli", "managed_identity", "default"])
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for GPT-5.4 (mutually exclusive with temperature)")
    parser.add_argument("--max_edge", type=int, default=1024,
                        help="Max image edge for resize before base64")
    parser.add_argument("--default_score", type=float, default=0.1,
                        help="Score assigned when VLM parse fails")
    parser.add_argument("--use_mask", action="store_true",
                        help="Only evaluate indices in a mask file")
    parser.add_argument("--mask_file", type=str, default=None,
                        help='JSON mask file with {"indices": {"simple": [...], ...}}')
    # ── HARD-CODED EVALUATION SETTINGS ─────────────────────────────────────────
    # The following two parameters are fixed to match the official SciForma paper
    # evaluation protocol. Do NOT override them in normal usage.
    #
    # --split_dims (default: True, always enabled)
    #   Evaluates each structural axis (Component / Arrow / Text) with a separate
    #   GPT call. This is critical: without split_dims, the Arrow score is
    #   inflated by ~0.25 because the model's attention is split across all axes
    #   in a single prompt, causing it to miss missing arrows.
    #   Paper scores (67.59% SciForma-Base, 69.51% SciForma-9B) were obtained
    #   with split_dims=True.
    #
    # --rubrics_suffix "" (default: empty, loads benchmark_*_rubrics.json)
    #   The official eval uses benchmark_{level}_rubrics.json (no suffix).
    #   A _final variant exists but was created after the paper eval was run;
    #   using _final gives slightly different inventory lists and incomparable scores.
    parser.add_argument("--split_dims", action="store_true", default=True,
                        help="[FIXED=True] Evaluate each axis with a separate GPT call. "
                             "Must be True to reproduce paper scores.")
    parser.add_argument("--rubrics_suffix", type=str, default="",
                        help="[FIXED=''] Suffix for rubrics JSON files. "
                             "Empty string loads benchmark_*_rubrics.json, "
                             "which matches the official paper evaluation.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N samples per difficulty")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = output_dir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    # Init client pool
    pool = ClientPool(
        deployment_name=args.deployment_name,
        auth_mode=args.auth,
    )

    # ── Load benchmark entries ─────────────────────────────────────────────────
    mode_str = "HF"
    gen_mode = "sequential"
    if args.hf_benchmark and not args.gt_base:
        logger.info(f"Loading benchmark from HuggingFace: {args.hf_benchmark}")
        entries = load_benchmark_entries_from_hf(
            hf_repo=args.hf_benchmark,
            gen_dir=args.gen_dir,
            levels=args.levels,
            new_gen=args.new_gen,
        )
    else:
        mode_str = "INTERNAL" if args.internal else "CLEAN"
        gen_mode = "new_gen (sequential idx)" if args.new_gen else "old_gen (old_local_idx)"
        logger.info(f"Loading benchmark entries ({mode_str} mode, {gen_mode}) ...")
        entries = load_benchmark_entries(
            args.benchmark_dir, args.gt_base, args.gen_dir, args.levels,
            internal=args.internal,
            new_gen=args.new_gen,
            use_mask=args.use_mask, mask_file=args.mask_file,
            rubrics_suffix=getattr(args, "rubrics_suffix", ""),
        )
    if args.max_samples is not None:
        filtered, seen = [], {}
        for e in entries:
            seen.setdefault(e["level"], 0)
            if seen[e["level"]] < args.max_samples:
                filtered.append(e); seen[e["level"]] += 1
        entries = filtered
    _level_counts = {l: sum(1 for e in entries if e["level"] == l) for l in args.levels}
    _lc_str = ", ".join(f"{l}: {_level_counts[l]}" for l in args.levels)
    logger.info(f"Loaded {len(entries)} entries ({_lc_str})")

    if not entries:
        logger.error("No valid entries found. Check paths.")
        sys.exit(1)

    # Auto-compute workers: 4 concurrent requests per endpoint
    workers = args.workers if args.workers > 0 else 4 * pool.size
    if args.split_dims:
        # In split_dims mode, each sample spawns 3 internal threads,
        # so reduce outer workers to avoid overwhelming endpoints.
        workers = max(4, workers // 3)
        logger.info(f"Split-dims mode: {workers} outer workers × 3 dim threads")
    else:
        logger.info(f"Workers: {workers}  (4 x {pool.size} endpoints)")

    # Run evaluation
    per_sample_summaries = run_evaluation(
        entries, pool,
        workers=workers,
        max_edge=args.max_edge,
        num_retest=args.num_retest,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        split_dims=args.split_dims,
    )

    # Write per-sample JSONs
    for sample in per_sample_summaries:
        sample_file = per_sample_dir / f"{sample['level']}_{sample['local_idx']:05d}.json"
        with open(sample_file, "w", encoding="utf-8") as fp:
            json.dump(sample, fp, ensure_ascii=False, indent=2)

    # Aggregate
    all_scores = [s["avg_score"] for s in per_sample_summaries]
    overall_mean = round(float(np.mean(all_scores)), 4) if all_scores else 0.0

    level_means = {}
    for level in args.levels:
        level_scores = [s["avg_score"] for s in per_sample_summaries if s["level"] == level]
        level_means[level] = {
            "mean": round(float(np.mean(level_scores)), 4) if level_scores else 0.0,
            "count": len(level_scores),
            "std": round(float(np.std(level_scores)), 4) if level_scores else 0.0,
        }

    aspect_means = {}
    for aspect in ("component_score", "arrow_score", "text_score"):
        vals = [s[aspect] for s in per_sample_summaries if s.get(aspect) is not None]
        aspect_means[aspect] = round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "model": os.path.basename(args.gen_dir),
        "gen_dir": args.gen_dir,
        "gt_base": args.gt_base,
        "vlm_model": f"Azure OpenAI {args.deployment_name}",
        "mode": mode_str,
        "num_retest": args.num_retest,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "workers": workers,
        "num_endpoints": pool.size,
        "num_endpoints_total": pool.total,
        "total_entries": len(entries),
        "overall_mean": overall_mean,
        "overall_std": round(float(np.std(all_scores)), 4) if all_scores else 0.0,
        "level_means": level_means,
        "aspect_means": aspect_means,
        "levels": args.levels,
    }

    summary_file = output_dir / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"  EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Model          : {summary['model']}")
    logger.info(f"  VLM            : {args.deployment_name} ({pool.size} endpoints)")
    logger.info(f"  Mode           : {mode_str}")
    logger.info(f"  Total entries  : {len(entries)}")
    logger.info(f"  Num retest     : {args.num_retest}")
    logger.info(f"  Temperature    : {args.temperature}")
    logger.info(f"  Max tokens     : {args.max_completion_tokens}")
    logger.info(f"  Workers        : {workers}")
    logger.info(f"  Overall mean   : {overall_mean:.4f}")
    for level, lm in level_means.items():
        logger.info(f"  {level:8s} mean : {lm['mean']:.4f} +/- {lm['std']:.4f}  (n={lm['count']})")
    logger.info(f"  Aspect means   : {aspect_means}")
    logger.info(f"  Summary file   : {summary_file}")
    logger.info(f"  Per-sample dir : {per_sample_dir}")


if __name__ == "__main__":
    main()
