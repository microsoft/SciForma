"""Refine an editable SciForma prompt from user feedback."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from generate.utils.llm_client import chat

SYSTEM_PROMPT = """You are revising a prompt for a scientific diagram diffusion model.
Return only the complete revised prompt, with no preamble or markdown.

Preserve correct technical content, module names, mathematical symbols, and intended data flow. Apply the user's feedback precisely. Remove ambiguous, duplicated, contradictory, or hallucinated connections. Express each arrow once with an explicit source and destination. Keep labels short and readable. Keep the global layout feasible and avoid unnecessary visual decoration."""

DEFAULT_FEEDBACK = """Audit and improve this prompt for structural clarity. Remove duplicated components and ambiguous arrows, make every directed connection explicit, keep labels readable, and preserve all technical content."""

DEFAULT_VISION_FEEDBACK = """Make the figure more publication-ready while preserving the scientific story. Prefer a compact, balanced composition, semantic graphics instead of explanatory text, short labels, clear primary flow, and narrow uniform outer margins."""

VISION_REVIEW_SYSTEM_PROMPT = """You are a meticulous scientific-figure art director and prompt editor.
Compare the generated image with the target request, approved plan, and current diffusion prompt. Think through the review internally before answering.

Return exactly one JSON object with these two string fields and no markdown fences:
{
  "critique": "A concise, concrete review grouped inline by Structure, Arrows, Text, Color, and Readability.",
  "revised_prompt": "The complete replacement diffusion prompt."
}

The critique must describe visible problems, not generic advice. In the revised prompt:
- Preserve all correct technical content, mathematical symbols, module names, and intended data flow. Do not invent scientific claims.
- Follow the user's high-level revision instruction unless it conflicts with scientific truth or the supplied target.
- Make visual edits rather than merely expanding prose. Merge subordinate checks or repeated operations into coherent groups, and replace suitable text-only sub-boxes with meaningful technical graphics such as document stacks, clustered dots, image mosaics, funnels, reviewer silhouettes, node-link graphs, or feature tensors.
- Define a feasible global layout and every major component's relative position and shape. Center one compact composition, fill the canvas evenly, and keep narrow visually equal outer margins on all four sides. Remove large empty top/bottom bands and isolated floating modules; keep every label, object, connector, and arrowhead fully inside the frame without cropping.
- Define every essential arrow exactly once with an explicit source and destination. State one consistent global arrow style, then mention a different line type only when it encodes a real distinction; remove duplicated, dangling, crossing, or contradictory connections.
- Put double quotation marks around every English label that should be visibly rendered in the diagram. Keep visible labels short and do not quote prose that is only an instruction to the renderer.
- Preserve every color or palette explicitly supplied by the user; it overrides all defaults. If the user did not choose colors, diagnose the current harmony and select 3-5 plain-name colors appropriate to the domain. Use a restrained complementary pair, an analogous family with one warm accent, or a balanced categorical palette as appropriate. Balance warm and cool areas, reserve the strongest accent for the key result, and never default every module to blue/teal. Keep text, outlines, and arrows dark and high-contrast; avoid neon saturation, rainbow palettes, gradients, and generic PowerPoint styling.
- Never leave a "visual demo", "sample image", "thumbnail", "mosaic", "example output", photograph, miniature graph, or plot underspecified. The revised prompt must state its concrete visible subject/data, principal object count and arrangement, important shapes and colors, internal marks or links, and the scientific contrast it demonstrates. If the source provides no concrete scene, use a nonfigurative technical miniature and do not invent people, faces, animals, logos, or photorealistic scenes. Default miniature graphs to simple geometric nodes and straight thin edges; reject anatomy-like curves or creature silhouettes.
- Preserve one canonical definition for every repeated visual asset and require identical reuse beneath task-specific overlays. Unless the user requests otherwise, use a uniform white or very light neutral-gray background; stage zones must be flat regular rectangles. Remove painterly textures, brush strokes, translucent washes, amorphous blobs, organic swashes, and irregular decorative background shapes. Functional irregular geometry is allowed only inside a bounded scientific inset.
- Replace every bare "document icon" or "paper icon" with flat primitives: two or three slightly offset upright white rectangular sheets with thin dark straight outlines; one small upper-right triangular fold and exactly three short parallel gray line marks on the front sheet. Do not add handwriting, photos, logos, torn or curled paper, or tiny paragraphs.
- Specify a clean publication-ready scientific-paper style with aligned groups, purposeful internal spacing, consistent stroke widths, high contrast, and readable typography. Do not request generic "generous whitespace" because it can create uneven empty areas.
- Do not increase the number of major groups, primary arrows, or visible labels unless the user explicitly asks for missing content. Keep only decisive numeric annotations and never put a sentence or checklist inside a node.
- End with a short negative constraint that forbids duplicate labels/modules, illegible or misspelled text, ambiguous arrows, cropped elements, uneven outer whitespace, meaningless decoration, and visual clutter.

Before returning the JSON, silently verify that all planned modules appear once, every connection has endpoints, all visible English labels are double-quoted, and the palette is consistent."""


def _normalise_json_value(value: Any) -> str:
    """Convert a JSON field to a compact string without accepting opaque objects."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(item.strip() for item in value if item.strip())
    return ""


def _parse_vision_review(response: str, current_prompt: str) -> tuple[str, str]:
    """Parse a review JSON object, tolerating prose or a fenced JSON wrapper."""
    raw = (response or "").strip()
    if not raw:
        return (
            "Vision review returned an empty response; the current prompt was kept unchanged.",
            current_prompt,
        )

    candidates: list[Any] = [raw]
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())

    # JSONDecoder.raw_decode can recover a valid object after a short preamble.
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(value)
        break

    parsed: dict[str, Any] | None = None
    for candidate in candidates:
        value = candidate
        if isinstance(value, str):
            try:
                value = json.loads(value)
                # Some gateways/models double-encode the object as a JSON string.
                if isinstance(value, str):
                    value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(value, dict):
            parsed = value
            break

    if parsed is None:
        return (
            "Vision review could not be parsed safely; the current prompt was kept unchanged. Please run the review again.",
            current_prompt,
        )

    critique = _normalise_json_value(parsed.get("critique"))
    revised_prompt = _normalise_json_value(
        parsed.get("revised_prompt")
        or parsed.get("revisedPrompt")
        or parsed.get("prompt")
    )
    if not revised_prompt:
        return (
            critique
            or "Vision review did not return a revised prompt; the current prompt was kept unchanged.",
            current_prompt,
        )
    if not critique:
        critique = "The image was reviewed and the prompt was revised; no separate critique was returned."
    return critique, revised_prompt


async def refine_prompt(current_prompt: str, feedback: str, model: str = "gpt-5.4") -> str:
    request = feedback.strip() or DEFAULT_FEEDBACK
    return await chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Current SciForma prompt:\n{current_prompt}\n\n"
                    f"Revision request:\n{request}\n\n"
                    "Return the full revised prompt only."
                ),
            }
        ],
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=0.4,
    )


async def refine_prompt_from_image(
    current_prompt: str,
    image_path: str,
    target_request: str,
    approved_plan: str,
    model: str = "gpt-5.4",
) -> str:
    """Inspect a generated image and return a corrected full diffusion prompt.

    This remains the YOLO-mode compatibility wrapper. Human mode can call
    :func:`analyze_and_refine_prompt_from_image` to show the critique as well.
    """
    _, revised_prompt = await analyze_and_refine_prompt_from_image(
        current_prompt=current_prompt,
        image_path=image_path,
        target_request=target_request,
        approved_plan=approved_plan,
        model=model,
    )
    return revised_prompt


async def analyze_and_refine_prompt_from_image(
    current_prompt: str,
    image_path: str,
    target_request: str,
    approved_plan: str,
    model: str = "gpt-5.4",
    high_level_instruction: str = "",
) -> tuple[str, str]:
    """Return a concise visual critique and a complete replacement prompt."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Generated image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    revision_request = high_level_instruction.strip() or DEFAULT_VISION_FEEDBACK
    response = await chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Inspect this generated scientific diagram. Compare it carefully with the target request, "
                            "approved plan, and current prompt. Diagnose concrete visible issues in structure, arrows, "
                            "text, color, and readability, then produce a complete corrected diffusion prompt.\n\n"
                            f"User high-level revision instruction:\n{revision_request}\n\n"
                            f"Target request:\n{target_request}\n\n"
                            f"Approved plan:\n{approved_plan}\n\n"
                            f"Current prompt:\n{current_prompt}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"},
                    },
                ],
            }
        ],
        model=model,
        system=VISION_REVIEW_SYSTEM_PROMPT,
        max_tokens=8192,
        temperature=0.2,
    )
    return _parse_vision_review(response, current_prompt)
