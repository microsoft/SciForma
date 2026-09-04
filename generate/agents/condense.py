"""
agents/condense.py — Condense raw diagram description into FLUX-compatible prompt.

This is the key contribution: rewrites verbose planner output into the dense,
quoted-label style that SciForma was trained on (1800–2500 chars).
"""
from __future__ import annotations

import json
import logging

from generate.agents.icl_retriever import retrieve_candidates
from generate.utils.llm_client import chat

LOGGER = logging.getLogger("sciforma.condense")

SYSTEM_PROMPT = """\
You rewrite a raw scientific-diagram plan into a concise observer-style description of
the finished figure for SciForma. Use the offline-screened text references for observer-
style prose, semantic specificity, content density, and transferable composition/color
relationships. Never copy their scientific content.

SEMANTIC FIDELITY:
- Preserve the main modules, causal order, branches, defining variables or equations,
  and final outputs. Never invent, rename, reverse, or contradict scientific content.
- Technical fidelity is not literal transcription. Merge subordinate checks, repeated
  operations, and implementation detail into coherent groups while retaining the visual
  evidence needed to understand the method.
- Any user-provided layout, style, color, icon, or visual-example instruction is binding
  and has priority over references or defaults. Copy requested color names exactly.
- If the raw plan names concrete colors, preserve them. If it leaves hues unassigned,
  choose a coherent 3-5 color palette informed by the selected high-aesthetic references
  and the target domain. Vary the palette with the content rather than using one default;
  keep text and arrows dark and use shape as a second categorical cue.

OBSERVER STYLE AND LABELS:
- Write as an observer describing what is visible, not as a sequence of drawing commands.
- Put every visible English label in straight ASCII double quotes. Keep labels to 1-4
  words whenever possible; never place a prose sentence or checklist inside a node.
- Use concrete visual nouns: a document stack, tensor grid, clustered dots, filter funnel,
  graph, plot, or paired cards. Avoid generic boxes when a scientific visual can carry the
  meaning more clearly.

VISUAL EXAMPLE CONTRACT:
- Never leave a pictorial element underspecified. The phrases "visual demo", "sample
  image", "thumbnail", "image mosaic", "example output", and "icon" are incomplete
  unless followed by a concrete description of what is visibly inside.
- For every demo, inset, thumbnail, photograph, or miniature diagram, state its position
  and boundary, the visible subject or data, the number and arrangement of principal
  objects, their important shapes and colors, and the scientific contrast it demonstrates.
  Describe internal graph edges, plot marks, overlays, or before/after differences when
  they matter. This detail belongs in prompt prose, not as extra visible labels.
- Never compress a canonical visual-asset definition into "thumbnail", "icon", "small
  graph", or another generic substitute. Preserve its exact object counts, topology,
  colors, reuse constraints, and state-to-state changes even if the prompt becomes longer.
  Define the asset once, then explicitly call later occurrences the identical base asset.
- Never leave "document icon" or "paper icon" in the final prompt. Unless overridden by
  the user, describe it as two or three offset upright white rectangular sheets with thin
  dark straight outlines; give the front sheet one small upper-right triangular fold and
  exactly three short parallel gray line marks. A prompt card is one flat front sheet.
  Forbid handwriting, photos, logos, torn or curled edges, and tiny paragraphs.
- A miniature node-link graph must include an approximate node count, spatial arrangement,
  node shapes or colors, and edge pattern. Neighboring stages must not reuse the identical
  miniature unless it is intentionally the same data; otherwise state the visible edit,
  deletion, highlight, deformation, or annotation that distinguishes the two states. Use
  simple geometric nodes and straight thin edges unless the source says otherwise; forbid
  anatomy-like curves, faces, limbs, or organic creature silhouettes.
- If the source does not specify a concrete scene, use a simple nonfigurative technical
  miniature such as a node-link graph, chart, grid, document, or feature map. Do not invent
  people, faces, animals, branded logos, or photorealistic scenes. Prefer flat vector
  silhouettes to realistic characters when a human role is scientifically necessary.

LAYOUT AND CONNECTIONS:
- State one feasible reading direction, major groups, containment, and relative positions.
  Center a compact composition, use balanced gaps and narrow visually equal outer margins,
  and keep every object, label, connector, and arrowhead fully inside the canvas.
- Describe each essential edge exactly once with source, destination, direction, and any
  meaningful label. State the common arrow style once; specify a different line style only
  for a real distinction such as feedback, gradient, optional flow, or loss.
- Avoid overlap, crossing arrows, isolated floating modules, duplicated elements, large
  empty outer bands, decorative backgrounds, and cropped content. Unless the user asks
  otherwise, use a uniform white or very light neutral-gray background. Functional zones
  must be flat regular rectangles aligned to the grid. Forbid painterly textures, brush
  strokes, translucent washes, amorphous blobs, organic swashes, and irregular decorative
  background shapes; functional irregular geometry may appear only inside a bounded
  scientific inset.

OUTPUT:
Return only continuous English prose with no markdown, bullets, preamble, or figure title.
Open with "The figure illustrates...", then describe layout, components and their detailed
visual contents, followed by the essential connections. The 1800-2500 character range of
many training prompts is a soft density reference only. Use more when a visual example
requires precise description, and never truncate a complete prompt to meet a character
target. Preserve all aesthetic choices established by the user and upstream stylist;
never standardize them into a default palette."""

MAX_CHARS = 4800

ICL_SELECTOR_PROMPT = """Select the two benchmark prompts whose visual composition is
most useful as writing-style references for the requested scientific diagram. The
available candidates have already passed offline GT-image aesthetics and text-transfer
checks and moderate prompt length. Their aspect ratio is within ±10% when possible;
otherwise they are the nearest high-quality references. Now prioritize semantic
method similarity and the same topology (linear, parallel,
split/merge, feedback, multi-panel), then low text density, clear hierarchy, meaningful
visual metaphors, detailed visual-demo descriptions, balanced composition, harmonious
palette relationships, and reading direction. Prefer observer-style references that
communicate through graphics rather than paragraphs or many text boxes. Do not copy
scientific content between examples. Return exactly one JSON object:
{"selected": ["split:index", "split:index"]}
Use only candidate IDs and select two distinct entries."""


def _excerpt(text: str, limit: int = 1800) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 25) // 2
    return text[:half] + "\n...[middle omitted]...\n" + text[-half:]


def _visual_budget(width: int, height: int) -> str:
    ratio = width / height
    if ratio >= 2.7:
        groups, labels, numbers, edges = 7, 12, 5, 9
    elif ratio >= 1.4:
        groups, labels, numbers, edges = 7, 13, 4, 9
    else:
        groups, labels, numbers, edges = 6, 12, 4, 8
    return (
        f"Target canvas {width}x{height} (aspect {ratio:.2f}:1). Aim for no more than "
        f"{groups} major visual groups, {labels} visible labels, {numbers} numeric "
        f"annotations, and {edges} primary arrows when the scientific content allows it; "
        "do not omit an essential module or underspecify a visual demo merely to meet "
        "these advisory counts. Prefer semantic graphics over visible prose. "
        "Center one compact composition, fill the frame evenly, use narrow equal outer "
        "margins, and keep every element fully visible without cropping."
    )


def _parse_selected_ids(response: str, valid_ids: set[str]) -> list[str]:
    raw = (response or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        selected = payload.get("selected", []) if isinstance(payload, dict) else []
        result: list[str] = []
        for item in selected:
            if isinstance(item, str) and item in valid_ids and item not in result:
                result.append(item)
        if len(result) >= 2:
            return result[:2]
    return []


async def _select_icl_examples(
    raw_description: str,
    model: str,
    target_width: int,
    target_height: int,
    n: int = 2,
) -> list[dict]:
    shortlist = retrieve_candidates(
        raw_description,
        width=target_width,
        height=target_height,
        limit=10,
    )
    if len(shortlist) <= n:
        return shortlist

    candidate_block = "\n\n".join(
        (
            f"Candidate {item['id']} | {item['width']}x{item['height']} | "
            f"aesthetic {item['aesthetic_score']:.0f} | transfer "
            f"{item['transfer_score']:.0f} | retrieval {item['score']:.3f}\n"
            f"{_excerpt(item['prompt'])}"
        )
        for item in shortlist
    )
    selector_message = (
        f"Target resolution: {target_width}x{target_height} "
        f"(aspect ratio {target_width / target_height:.3f})\n\n"
        f"Current planned diagram:\n{_excerpt(raw_description, 5000)}\n\n"
        f"Shortlisted local benchmark prompts:\n{candidate_block}"
    )
    response = await chat(
        messages=[{"role": "user", "content": selector_message}],
        model=model,
        system=ICL_SELECTOR_PROMPT,
        max_tokens=1024,
        temperature=0.0,
    )
    selected_ids = _parse_selected_ids(response, {item["id"] for item in shortlist})
    if not selected_ids:
        return shortlist[:n]
    by_id = {item["id"]: item for item in shortlist}
    return [by_id[item_id] for item_id in selected_ids[:n]]


async def condense(
    raw_description: str,
    model: str = "gpt-5.4",
    target_width: int = 1024,
    target_height: int = 1024,
    selected_icl: list[dict] | None = None,
) -> str:
    """
    Rewrite a raw planner description into the SciForma FLUX prompt style.

    Args:
        raw_description: Output from PlannerAgent
        model: LLM model or Azure deployment name
        target_width: Requested output width used for ICL resolution matching
        target_height: Requested output height used for ICL resolution matching

    Returns:
        Condensed prompt ready for SciForma-9B image generation
    """
    icl = selected_icl
    if icl is None:
        icl = await _select_icl_examples(
            raw_description,
            model=model,
            target_width=int(target_width),
            target_height=int(target_height),
            n=2,
        )
    LOGGER.info("Selected ICL references: %s", [item["id"] for item in icl])
    icl_block = ""
    for i, item in enumerate(icl, 1):
        reference = item["prompt"]
        icl_block += (
            f"\n=== Dynamically selected reference {i}: {item['id']} | "
            f"{item['width']}x{item['height']} ({len(reference)} chars) ===\n"
            f"{reference}\n"
        )

    visual_budget = _visual_budget(int(target_width), int(target_height))
    user_msg = (
        f"=== Canvas and readability guidance ===\n{visual_budget}\n"
        f"{icl_block}\n"
        f"=== Raw planner description to rewrite ({len(raw_description)} chars) ===\n"
        f"{raw_description}\n\n"
        f"=== Rewritten description (normally <={MAX_CHARS} chars; clarity and completeness "
        f"come first; use exact straight double-quoted visible labels, explicit edge "
        f"geometry, and reference density only) ==="
    )

    candidate = await chat(
        messages=[{"role": "user", "content": user_msg}],
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=1.0,
    )

    # Never truncate by character count: the model's 2048-token text-encoder budget is
    # much larger than 4800 English characters, and the explicit edge list comes last.
    result = candidate.strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result
