"""Retrieve resolution- and structure-matched ICL prompts from SciFormaBench."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "eval" / "prompts"
AESTHETIC_MANIFEST = REPO_ROOT / "generate" / "data" / "sciformabench_aesthetic_text_icl.json"
SPLITS = ("simple", "medium", "hard")
ASPECT_RATIO_TOLERANCE = 0.10
MIN_REFERENCE_CHARS = 1600
MAX_REFERENCE_CHARS = 4800
TARGET_REFERENCE_CHARS = 2800

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
_STOP_WORDS = {
    "about", "above", "after", "again", "against", "also", "among", "another",
    "around", "because", "before", "below", "between", "both", "contains", "depicted",
    "describe", "diagram", "each", "entire", "figure", "first", "following", "from",
    "global", "illustrates", "inside", "into", "labeled", "layout", "left", "main",
    "next", "only", "other", "output", "place", "placed", "presents", "represented",
    "right", "second", "shown", "shows", "stage", "than", "that", "their", "then",
    "there", "these", "third", "this", "through", "using", "where", "which", "while",
    "with", "within",
}
_TOPOLOGY_TERMS = {
    "arrow", "arrows", "branch", "branches", "cyclic", "cycle", "dashed", "decision",
    "feedback", "flow", "graph", "hierarchy", "horizontal", "loss", "merge", "multi-stage",
    "panel", "panels", "parallel", "pipeline", "sequential", "solid", "split", "timeline",
    "tree", "vertical",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if token not in _STOP_WORDS
    ]


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[dict], dict[str, float], float]:
    manifest = json.loads(AESTHETIC_MANIFEST.read_text(encoding="utf-8"))
    aesthetic_references = manifest.get("references", {})
    records: list[dict] = []
    document_frequency: Counter[str] = Counter()
    total_terms = 0

    for split in SPLITS:
        payload = json.loads((PROMPT_DIR / f"{split}.json").read_text())
        prompts = payload["validation_prompts"]
        resolutions = payload["resolution_list"]
        if len(prompts) != len(resolutions):
            raise ValueError(f"Mismatched prompts and resolutions in {split}.json")
        for index, (prompt, resolution) in enumerate(zip(prompts, resolutions)):
            reference_id = f"{split}:{index}"
            scores = aesthetic_references.get(reference_id)
            if not isinstance(scores, list) or len(scores) != 2:
                continue
            width, height = (int(resolution[0]), int(resolution[1]))
            terms = _tokens(prompt)
            counts = Counter(terms)
            document_frequency.update(counts.keys())
            total_terms += len(terms)
            records.append(
                {
                    "id": reference_id,
                    "split": split,
                    "index": index,
                    "width": width,
                    "height": height,
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                    "term_counts": counts,
                    "term_count": len(terms),
                    "topology": frozenset(counts).intersection(_TOPOLOGY_TERMS),
                    "aesthetic_score": float(scores[0]),
                    "transfer_score": float(scores[1]),
                }
            )

    count = len(records)
    inverse_document_frequency = {
        term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    average_length = total_terms / count
    return records, inverse_document_frequency, average_length


def _bm25(
    query_terms: set[str],
    record: dict,
    inverse_document_frequency: dict[str, float],
    average_length: float,
) -> float:
    k1, b = 1.5, 0.72
    length_ratio = record["term_count"] / average_length if average_length else 1.0
    score = 0.0
    for term in query_terms:
        frequency = record["term_counts"].get(term, 0)
        if not frequency:
            continue
        denominator = frequency + k1 * (1.0 - b + b * length_ratio)
        score += inverse_document_frequency.get(term, 0.0) * frequency * (k1 + 1.0) / denominator
    return score


def retrieve_candidates(
    query: str,
    width: int,
    height: int,
    limit: int = 8,
) -> list[dict]:
    """Return deterministic candidates matched by resolution, semantics, and topology."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if limit <= 0:
        return []

    records, inverse_document_frequency, average_length = _load_index()
    query_terms = set(_tokens(query))
    query_topology = query_terms.intersection(_TOPOLOGY_TERMS)
    target_ratio = width / height

    moderate_records = [
        record
        for record in records
        if MIN_REFERENCE_CHARS <= record["prompt_chars"] <= MAX_REFERENCE_CHARS
    ]
    resolution_ranked: list[tuple[float, float, dict]] = []
    for record in moderate_records:
        aspect_distance = abs(math.log((record["width"] / record["height"]) / target_ratio))
        aspect_relative_error = abs((record["width"] / record["height"]) / target_ratio - 1.0)
        size_distance = 0.5 * (
            abs(math.log(record["width"] / width))
            + abs(math.log(record["height"] / height))
        )
        resolution_ranked.append(
            (aspect_relative_error, aspect_distance + 0.1 * size_distance, record)
        )

    # Prefer a strict multiplicative ±10% aspect-ratio band. If that band is empty,
    # fall back to the nearest high-aesthetic text references as requested.
    resolution_ranked.sort(key=lambda item: (item[0], item[1], item[2]["id"]))
    in_band = [
        item for item in resolution_ranked if item[0] <= ASPECT_RATIO_TOLERANCE
    ]
    pool = list(in_band) if in_band else resolution_ranked[:limit]

    scored: list[dict] = []
    bm25_values = [
        _bm25(query_terms, record, inverse_document_frequency, average_length)
        for _, _, record in pool
    ]
    maximum_bm25 = max(bm25_values, default=1.0) or 1.0
    for (aspect_relative_error, combined_distance, record), bm25_value in zip(pool, bm25_values):
        semantic_score = bm25_value / maximum_bm25
        resolution_score = max(
            0.0,
            1.0 - aspect_relative_error / ASPECT_RATIO_TOLERANCE,
        )
        union = query_topology.union(record["topology"])
        topology_score = (
            len(query_topology.intersection(record["topology"])) / len(union)
            if union
            else 0.0
        )
        length_score = math.exp(
            -abs(record["prompt_chars"] - TARGET_REFERENCE_CHARS) / 1800.0
        )
        quality_score = min(
            max(0.0, (record["aesthetic_score"] - 82.0) / 9.0),
            max(0.0, (record["transfer_score"] - 82.0) / 13.0),
        )
        total_score = (
            0.57 * semantic_score
            + 0.18 * topology_score
            + 0.15 * resolution_score
            + 0.05 * length_score
            + 0.05 * quality_score
        )
        scored.append(
            {
                "id": record["id"],
                "split": record["split"],
                "index": record["index"],
                "width": record["width"],
                "height": record["height"],
                "prompt": record["prompt"],
                "score": total_score,
                "semantic_score": semantic_score,
                "resolution_score": resolution_score,
                "aspect_relative_error": aspect_relative_error,
                "aspect_distance": combined_distance,
                "prompt_chars": record["prompt_chars"],
                "aesthetic_score": record["aesthetic_score"],
                "transfer_score": record["transfer_score"],
            }
        )

    scored.sort(key=lambda item: (-item["score"], item["aspect_distance"], item["id"]))
    return scored[:limit]
