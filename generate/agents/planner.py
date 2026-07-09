"""
agents/planner.py — Diagram Planner Agent (no retriever, LaTeX-native input).

Takes methodology text + figure caption → returns detailed diagram description.
Uses OpenAI/Azure via llm_client.
"""
from __future__ import annotations
from generate.utils.llm_client import chat

SYSTEM_PROMPT = """I am working on a task: given the 'Methodology' section of a paper and the caption of the desired figure, automatically generate a corresponding illustrative diagram description.

Your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

IMPORTANT:
- Describe each element and their connections in detail
- Include layout direction (left-to-right, top-to-bottom, etc.)
- Specify shapes, colors, and line styles
- Describe arrows and data-flow directions
- Include sub-panels if the figure has multiple parts
- Do NOT include figure titles in your description
- Be thorough — vague descriptions produce poor figures"""


async def plan(
    methodology_text: str,
    figure_caption: str,
    model: str = "gpt-4o",
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
    user_msg = (
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
