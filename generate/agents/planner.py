"""
agents/planner.py — Diagram Planner Agent (no retriever, LaTeX-native input).

Takes methodology text + figure caption → returns detailed diagram description.
Uses OpenAI/Azure via llm_client.
"""
from __future__ import annotations

from generate.utils.llm_client import chat

SYSTEM_PROMPT = """You are the semantic and visual-abstraction planner for a scientific
methodology diagram. Given relevant paper text and the target caption, select the one
clear visual story that communicates the method. Preserve the causal method faithfully;
do not transcribe the paper into boxes.

Plan only information that belongs in the target figure:
- State one feasible reading direction and the major panel or branch arrangement.
- Build 5-9 top-level visual groups, combining related modules, filters, repeated stages,
  datasets, and annotations into coherent units.
- Give a group-level directed edge list with exactly one source and one destination,
  including branch conditions, feedback, conditioning, or loss links when required.
- Preserve supplied names, symbols, counts, and causal direction. Do not invent claims.
- Treat any user-supplied visual instruction as binding. Preserve requested colors,
  palette, style, layout, and named visual examples verbatim in the plan; never replace
  an explicit user color with a default palette.
- If the source contains conflicting approximate and exact counts, keep both as clearly
  scoped annotations rather than implying invalid arithmetic.

Plan for visual abstraction and legibility:
- Use 5-9 top-level groups and one dominant reading path. Put related filters or
  operations inside one visual group rather than making every action a full-size node.
- Collapse homogeneous repeated stages into ranges or a representative block with ×N
  when this preserves the method, while retaining distinct operations within the group.
- Summarize long enumerations by category and count; include individual members only
  when the caption explicitly makes every member visually essential. Do not copy a
  long member list back into Required annotations; use the group name, count, and at
  most two representative examples instead.
- Keep background context, alternative models, implementation trivia, and derivations
  out of the main flow. They may be a short note only if necessary for interpretation.
- Avoid repeated labels, redundant edges, crossing connections, and paragraphs inside
  boxes. Route a secondary conditioning or feedback path as one shared bus when possible.
- Prefer meaningful visual representations over text: document stacks for prompts,
  clustered points for grouping, thumbnail mosaics for image collections, a globe for
  web retrieval, a funnel/check for filtering, a person/check for human review, paired
  cards for image-text data, node-link graphs for structured knowledge, and compact
  tensors/feature maps for neural stages. Use an icon only when it replaces explanatory
  text; never add decorative clip art.
- Never propose a vague "visual demo", "sample image", "thumbnail", "mosaic", or
  "example output" by itself. For every pictorial inset, specify the visible subject,
  object count and arrangement, important shapes and colors, and what scientific fact it
  demonstrates. If the paper does not specify concrete content, use a simple abstract
  technical miniature such as a chart, node-link graph, grid, or document card; do not
  invent people, faces, animals, branded logos, or photorealistic scenes.
- Limit visible text to short names and decisive quantities. Normally propose no more
  than 12-16 visible labels, 3-5 numeric annotations, and 8-12 group-level arrows.
- Classify source details internally as essential visual structure, supporting visual
  annotation, or implicit context. Do not make implicit context visible text.

Do not prescribe shadows, glow, exact pixel sizes, or typography. Do not include a figure title. Return a
structured, concise plan with the sections Global layout, Visual groups, Group-level
connections, and Visible labels. Aim for roughly 2200-3800 characters, but prioritize
an immediately understandable visual hierarchy over a hard length."""


async def plan(
    methodology_text: str,
    figure_caption: str,
    model: str = "gpt-5.4",
    target_width: int | None = None,
    target_height: int | None = None,
) -> str:
    """
    Generate a detailed diagram description from methodology text and caption.

    Args:
        methodology_text: Extracted methodology section from LaTeX
        figure_caption: The figure caption (\\caption{...} text)
        model: LLM model name

    Returns:
        Detailed diagram description string
    """
    canvas = ""
    if target_width and target_height:
        canvas = (
            f"Target canvas: {target_width}x{target_height} "
            f"(aspect ratio {target_width / target_height:.2f}:1).\n\n"
        )
    user_msg = (
        canvas
        +
        f"Methodology Section:\n{methodology_text}\n\n"
        f"Diagram Caption:\n{figure_caption}\n\n"
        "Now provide a detailed description of the figure to be generated "
        "(do not include figure titles):"
    )
    return await chat(
        messages=[{"role": "user", "content": user_msg}],
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=1.0,
    )
