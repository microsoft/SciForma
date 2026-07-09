"""
agents/condense.py — Condense raw diagram description into FLUX-compatible prompt.

This is the key contribution: rewrites verbose planner output into the dense,
quoted-label style that SciForma was trained on (1800–2500 chars).
"""
from __future__ import annotations
import json, random
from pathlib import Path
from generate.utils.llm_client import chat

_DATA_DIR = Path(__file__).parent.parent / "data"

SYSTEM_PROMPT = """\
You rewrite a raw diagram description into a concise visual scene description.

PRIORITY — PRESERVE ALL TECHNICAL CONTENT:
Your main job is to keep every technical detail from the raw description.
  - Every module name, layer type, and operation mentioned
  - Every data-flow path — what connects to what, in which direction
  - Mathematical symbols and formulas: x_t, L_cls, f(x), τ, ρ, σ
  - Sub-panels (a), (b), (c) — describe each fully
  - DO NOT drop modules to save space. All components matter.

FORMATTING — QUOTED LABELS:
When you name a component, wrap its short name in single quotes.
  WRONG: a blue box representing the encoder
  RIGHT: a blue rounded box labeled 'Encoder'
Mathematical variables are labels too: 'x_t', 'L_cls', 'q_φ'.
Typically 15-25 quoted labels is enough for a full diagram.
Do NOT repeat the same label more than once — name each component once,
then refer to it by its description or position.

STYLE — write as an observer looking at the finished diagram:
  WRONG: Create a rounded rectangle with fill #EAF4FF and 2px border
  RIGHT: a blue rounded box labeled 'State Enc' sits at the left

COLOURS — use plain names only:
  blue, red, green, orange, teal, pink, purple, gold, gray, cyan, coral,
  light blue, light green, dark green, dark blue
  NEVER use: hex codes, "pastel", "pale", "muted", "soft", "faint",
  "subtle", "vivid", "neon", CSS syntax, pixel sizes, font names, emoji.
  NEVER describe the background. NEVER say "clean white background".

FORMAT: continuous prose, no markdown, no bullets. Open with
"The figure illustrates..." then global layout, then each component
with shape + colour + 'label' + role, then each arrow.
LENGTH: 1800-2500 characters. Be CONCISE. Every word must carry content.
Do not pad with visual fluff like "ample spacing", "tidy layout", "crisp
lines", "faint haze". These waste char budget and produce no visual detail.

EXAMPLE of the target style (1262 chars — note how dense and content-rich):
\"\"\"This figure illustrates two core components of the 4DGS-1K method:
transient Gaussian pruning and temporal filtering. The left panel (a)
shows a 3D spatial-temporal visualization where Gaussians evolve over
time along the X-axis (spatial) and t-axis (temporal). A red Gaussian,
marked with an 'x', represents a pruned Gaussian due to short lifespan;
it is visually distinct with a sharp, narrow peak and is surrounded by
other Gaussians (yellow/orange) that persist longer. Training views are
indicated by blue triangles, while testing views are black triangles.
The right panel (b) demonstrates the temporal filter mechanism: at time
t₀, a cube contains multiple Gaussians (colored yellow, orange, gray,
and red), representing active and inactive states. An arrow labeled
'Filter' points to a subsequent cube at time t₀+Δₜ which contains only
the active Gaussians (yellow and orange) with inactive ones removed.\"\"\"

I will give you 2 more reference examples. Match their content density."""

MAX_CHARS = 2800


def _pick_icl_examples(target_len: int = 2200, n: int = 2) -> list[str]:
    golden_path = _DATA_DIR / "golden_prompts.json"
    if not golden_path.exists():
        return []
    pool = json.loads(golden_path.read_text())
    scored = sorted(pool, key=lambda x: abs(x.get("golden_prompt_chars", 0) - target_len))
    chosen = scored[:n * 3]
    random.shuffle(chosen)
    return [x["golden_prompt"] for x in chosen[:n] if x.get("golden_prompt")]


async def condense(
    raw_description: str,
    model: str = "gpt-4o",
) -> str:
    """
    Rewrite a raw planner description into the SciForma FLUX prompt style.

    Args:
        raw_description: Output from PlannerAgent
        model: LLM model (gpt-4o, o3, etc.)

    Returns:
        Condensed prompt ready for SciForma-9B image generation
    """
    icl = _pick_icl_examples(target_len=2200, n=2)
    icl_block = ""
    for i, ref in enumerate(icl, 1):
        icl_block += f"\n=== Reference example {i} ({len(ref)} chars) ===\n{ref}\n"

    user_msg = (
        f"{icl_block}\n"
        f"=== Raw planner description to rewrite ({len(raw_description)} chars) ===\n"
        f"{raw_description}\n\n"
        f"=== Rewritten description (target ~2200 chars, use quoted labels, "
        f"match reference density) ==="
    )

    result = await chat(
        messages=[{"role": "user", "content": user_msg}],
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=1.0,
    )

    if len(result) > MAX_CHARS:
        cut = result[:MAX_CHARS]
        last_period = cut.rfind(".")
        if last_period > MAX_CHARS * 0.7:
            result = cut[:last_period + 1]
        else:
            result = cut

    return result
