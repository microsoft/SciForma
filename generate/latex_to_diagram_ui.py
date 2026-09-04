#!/usr/bin/env python3
"""Minimal local UI for the SciForma LaTeX-to-Diagram Agent."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from generate.agents.condense import condense
from generate.agents.planner import plan
from generate.agents.refine import (
    analyze_and_refine_prompt_from_image,
    refine_prompt_from_image,
)
from generate.agents.stylist import style

PUBLIC_MODEL = "LoYuXrqw/SciForma-9B"
PUBLIC_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4").strip() or "gpt-5.4"
VISION_MODEL = os.environ.get("VISION_MODEL", LLM_MODEL).strip() or LLM_MODEL
TESTED_GRADIO_VERSION = "5.50.0"
AUTO_TARGET = (
    "Create one clear methodology diagram that best represents the architecture, "
    "pipeline, or process described in the supplied paper context."
)
MODEL_REVISIONS = {
    "LoYuXrqw/SciForma-9B": "70cc9b0665681ec63a02f3067253481c9dc75184",
    "LoYuXrqw/SciForma-Base": "c19617cb11c755ee2f1d9a1390ff937e8b56c70a",
}
BASE_MODEL_REVISIONS = {
    "black-forest-labs/FLUX.2-klein-base-9B": "32773329fbe7e81a90ef971740e8ba4b0364ecf3",
}

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()
_MODEL_DIRS = None
_DOWNLOAD_LOCK = threading.Lock()
LOGGER = logging.getLogger("sciforma.ui")
_SESSION_JOBS: dict[str, dict] = {}
_SESSION_CURRENT: dict[str, str] = {}
_SESSION_JOBS_LOCK = threading.Lock()
_RESULT_TTL_SECONDS = 30 * 60


class _RenderCancelled(RuntimeError):
    """Internal signal raised from the diffusion step callback."""


MODEL_DOWNLOADS = (
    {
        "repo_id": PUBLIC_MODEL,
        "revision": MODEL_REVISIONS[PUBLIC_MODEL],
        "allow_patterns": ("transformer/*",),
        "ignore_patterns": (),
        "token": None,
    },
    {
        "repo_id": PUBLIC_BASE_MODEL,
        "revision": BASE_MODEL_REVISIONS[PUBLIC_BASE_MODEL],
        "allow_patterns": (),
        # The top-level 18.2 GB file is a single-file export of the base
        # transformer.  The pipeline receives SciForma's transformer instead,
        # so neither it nor the transformer/ directory is needed here.
        "ignore_patterns": (
            "transformer/*",
            "flux-2-klein-base-9b.safetensors",
            "*.md",
            "*.jpg",
            "*.png",
        ),
        "token": "HF_TOKEN",
    },
)


def _load_1024_buckets() -> list[tuple[int, int]]:
    """Load canonical (width, height) buckets without importing training dependencies."""
    source = REPO_ROOT / "sciforma" / "datasets" / "ar_batch_sampler.py"
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "ASPECT_RATIO_1024"
            for target in node.targets
        ):
            values = ast.literal_eval(node.value)
            buckets = [(int(width), int(height)) for height, width in values.values()]
            return sorted(buckets, key=lambda item: item[0] / item[1])
    raise RuntimeError("ASPECT_RATIO_1024 was not found")


ASPECT_BUCKETS = _load_1024_buckets()


def _ratio_label(width: int, height: int) -> str:
    if width == height:
        ratio = "1:1"
    elif width < height:
        ratio = f"1:{height / width:.2f}".rstrip("0").rstrip(".")
    else:
        ratio = f"{width / height:.2f}".rstrip("0").rstrip(".") + ":1"
    return f"{ratio}  ·  {width} × {height}"


ASPECT_CHOICES = [(_ratio_label(width, height), f"{width}x{height}") for width, height in ASPECT_BUCKETS]
DEFAULT_RESOLUTION = "1024x1024"


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (AttributeError, TypeError, ValueError) as exc:
        raise gr.Error("Select a valid 1024 resolution bucket.") from exc
    if (width, height) not in ASPECT_BUCKETS:
        raise gr.Error("Select a valid 1024 resolution bucket.")
    return width, height

APP_CSS = """
:root { color-scheme: light; }
html, body { height: 100% !important; margin: 0 !important; overflow: hidden !important; }
body, .gradio-container { background: #f3f4f6 !important; color: #1f2937 !important; }
.gradio-container {
  width: 100vw !important; max-width: none !important; height: 100vh !important;
  margin: 0 !important; padding: 10px 12px !important; overflow: hidden !important;
  box-sizing: border-box !important;
}
.gradio-container > main {
  display: flex !important; flex-direction: column !important; height: 100% !important;
  min-height: 0 !important; max-height: 100% !important; overflow: hidden !important;
}
.sf-card {
  background: #ffffff !important; border: 1px solid #d1d5db !important;
  border-radius: 8px !important; box-shadow: none !important;
}
.sf-main {
  display: grid !important; grid-template-columns: minmax(0, 3fr) minmax(0, 7fr) !important;
  align-items: stretch !important; gap: 10px !important; width: 100% !important;
  flex: 1 1 0 !important; height: auto !important; min-height: 0 !important;
  max-height: none !important; overflow: hidden !important;
}
.sf-left {
  width: 100% !important; max-width: 100% !important;
  min-width: 0 !important; min-height: 0 !important; gap: 10px !important;
  display: grid !important; grid-template-rows: minmax(300px, 1fr) auto !important;
  overflow: hidden !important;
}
.sf-right {
  width: 100% !important; max-width: 100% !important;
  min-width: 0 !important; min-height: 0 !important; gap: 10px !important;
  display: grid !important; grid-template-rows: minmax(280px, 2fr) minmax(0, 3fr) !important;
  overflow: hidden !important;
}
.sf-card { padding: 10px 12px !important; min-height: 0 !important; overflow: hidden !important; }
.sf-source { min-height: 0 !important; }
.sf-settings { min-height: 245px !important; }
.sf-draft {
  display: grid !important; grid-template-rows: auto minmax(0, 1fr) auto auto auto !important;
  gap: 6px !important; min-height: 0 !important;
}
.sf-review-panel {
  background: #f9fafb !important; border: 1px solid #cbd5e1 !important;
  border-radius: 6px !important; padding: 9px 10px !important;
  gap: 6px !important;
}
.sf-review-panel textarea {
  min-height: 68px !important; max-height: 100px !important;
  background: #ffffff !important;
}
.sf-review-heading { color: #111827 !important; font-size: 13px !important; }
.sf-preview-card {
  display: grid !important; grid-template-rows: auto minmax(0, 1fr) !important;
  min-height: 0 !important; height: 100% !important;
}
.sf-title h2, .sf-title h3 { color: #111827 !important; font-size: 15px !important; margin: 0 0 10px !important; }
.sf-note { color: #6b7280 !important; font-size: 12px !important; }
.sf-editor textarea {
  background: #f9fafb !important; border: 1px solid #cbd5e1 !important;
  border-radius: 5px !important; color: #111827 !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
  font-size: 13px !important; line-height: 1.55 !important;
}
.sf-editor textarea:focus {
  background: #ffffff !important; border-color: #64748b !important;
  box-shadow: 0 0 0 1px #64748b !important;
}
.sf-button, .sf-button button, button.sf-button, .gr-button-primary {
  background: #173a63 !important; border: 1px solid #173a63 !important;
  color: #ffffff !important; box-shadow: none !important;
}
.sf-button:hover, .sf-button button:hover, button.sf-button:hover, .gr-button-primary:hover {
  background: #0f2d4f !important; border-color: #0f2d4f !important;
}
input[type="checkbox"], input[type="radio"] { accent-color: #173a63 !important; }
.sf-status {
  background: #f9fafb !important; border-left: 4px solid #6b7280 !important;
  color: #374151 !important; padding: 5px 9px !important; max-height: 42px !important;
  overflow-y: auto !important;
}
.sf-source-context textarea { height: clamp(210px, 32vh, 320px) !important; min-height: 0 !important; }
.sf-target textarea { height: clamp(75px, 11vh, 110px) !important; min-height: 0 !important; }
.sf-draft-editor { height: 100% !important; min-height: 0 !important; overflow: hidden !important; }
.sf-draft-editor > label {
  display: flex !important; flex-direction: column !important; height: 100% !important;
  min-height: 0 !important;
}
.sf-draft-editor .input-container { flex: 1 1 0 !important; min-height: 0 !important; }
.sf-draft-editor textarea {
  height: 100% !important; min-height: 0 !important; max-height: none !important;
  overflow-y: auto !important;
}
.sf-preview { background: #ffffff !important; border: 1px solid #d1d5db !important; height: 100% !important; min-height: 0 !important; }
.sf-preview > div { height: 100% !important; min-height: 0 !important; }
footer { display: none !important; }
"""


def _read_context(pasted_text: str):
    context = pasted_text.strip()
    if not context:
        raise gr.Error("Paste the relevant paper context first.")
    return context, AUTO_TARGET


def _download_model_files(local_files_only: bool) -> tuple[str, str]:
    global _MODEL_DIRS

    with _DOWNLOAD_LOCK:
        if _MODEL_DIRS is not None:
            return _MODEL_DIRS
        from huggingface_hub import snapshot_download

        model_config, base_config = MODEL_DOWNLOADS
        model_dir = snapshot_download(
            model_config["repo_id"],
            revision=model_config["revision"],
            allow_patterns=list(model_config["allow_patterns"]),
            local_files_only=local_files_only,
        )
        base_dir = snapshot_download(
            base_config["repo_id"],
            revision=base_config["revision"],
            ignore_patterns=list(base_config["ignore_patterns"]),
            token=os.environ.get(base_config["token"]),
            local_files_only=local_files_only,
        )
        _validate_model_files(Path(model_dir), Path(base_dir))
        _MODEL_DIRS = (model_dir, base_dir)
        return _MODEL_DIRS


def _validate_weight_index(index_path: Path) -> None:
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Empty weight map: {index_path}")
    for filename in set(weight_map.values()):
        shard = index_path.parent / filename
        if not shard.is_file():
            raise FileNotFoundError(shard)


def _validate_model_files(model_dir: Path, base_dir: Path) -> None:
    _validate_weight_index(model_dir / "transformer/diffusion_pytorch_model.safetensors.index.json")
    _validate_weight_index(base_dir / "text_encoder/model.safetensors.index.json")
    required = [
        base_dir / "model_index.json",
        base_dir / "scheduler/scheduler_config.json",
        base_dir / "tokenizer/tokenizer.json",
        base_dir / "vae/config.json",
        base_dir / "vae/diffusion_pytorch_model.safetensors",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)


async def _prepare_pipeline(
    progress,
    progress_start: float = 0.04,
    progress_end: float = 0.38,
) -> None:
    if _PIPELINE is not None:
        progress(progress_end, desc="Model ready")
        return
    try:
        await asyncio.to_thread(_download_model_files, True)
    except Exception as exc:
        raise gr.Error(
            "Local model cache is incomplete. Stop the UI and run "
            "`python generate/prepare_ui_models.py`, then restart it. "
            "HF_TOKEN is needed only for that one-time download."
        ) from exc

    progress(progress_start, desc="Loading the prepared local model")
    load_task = asyncio.create_task(asyncio.to_thread(_load_pipeline))
    while not load_task.done():
        progress(progress_end, desc="Loading the prepared local model into GPU memory")
        await asyncio.sleep(1.0)
    try:
        await load_task
    except Exception as exc:
        raise gr.Error(f"Model loading failed: {exc}") from exc


def _load_pipeline():
    global _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

        model_dir, base_dir = _download_model_files(local_files_only=True)
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_dir,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            base_dir,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.transformer.eval()
        _PIPELINE = pipe
        return pipe


def _render(
    prompt: str,
    width: int,
    height: int,
    cfg: float,
    steps: int,
    seed: int,
    step_state: list[int] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    if int(width) % 16 or int(height) % 16:
        raise gr.Error("Width and height must be multiples of 16.")

    import torch

    pipe = _load_pipeline()

    def on_step_end(_pipe, step: int, _timestep, callback_kwargs):
        if cancel_event is not None and cancel_event.is_set():
            raise _RenderCancelled("Image generation cancelled")
        if step_state is not None:
            step_state[0] = int(step) + 1
        return callback_kwargs

    with _PIPELINE_LOCK, torch.inference_mode():
        image = pipe(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            max_sequence_length=2048,
            generator=torch.Generator(device="cuda").manual_seed(int(seed)),
            callback_on_step_end=on_step_end,
        ).images[0]
    output = Path(tempfile.mkdtemp(prefix="sciforma-ui-")) / "diagram.png"
    image.save(output)
    return str(output)


async def _render_with_progress(
    progress,
    prompt: str,
    width: int,
    height: int,
    cfg: float,
    steps: int,
    seed: int,
    progress_start: float,
    progress_end: float,
    label: str,
) -> str:
    step_state = [0]
    cancel_event = threading.Event()
    progress_record = getattr(progress, "record", None)
    if isinstance(progress_record, dict):
        progress_record["cancel_event"] = cancel_event
    render_task = asyncio.create_task(
        asyncio.to_thread(
            _render,
            prompt,
            width,
            height,
            cfg,
            steps,
            seed,
            step_state,
            cancel_event,
        )
    )
    try:
        while not render_task.done():
            completed = min(max(step_state[0], 0), int(steps))
            fraction = completed / int(steps) if int(steps) else 0.0
            value = progress_start + (progress_end - progress_start) * fraction
            progress(value, desc=f"{label} — step {completed}/{int(steps)}")
            await asyncio.sleep(0.5)
        image_path = await render_task
    except asyncio.CancelledError:
        cancel_event.set()

        def consume_cancelled_render(task: asyncio.Task) -> None:
            try:
                task.result()
            except (asyncio.CancelledError, _RenderCancelled):
                pass
            except Exception:
                LOGGER.exception("Cancelled render exited with an unexpected error")

        render_task.add_done_callback(consume_cancelled_render)
        raise
    except _RenderCancelled as exc:
        raise asyncio.CancelledError from exc
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {exc}") from exc
    progress(progress_end, desc=f"{label} — complete")
    return image_path


def reset_workflow():
    return (
        "start",
        "",
        "",
        "",
        gr.update(value="", label="Instruction"),
        None,
        gr.update(value="Start", interactive=True),
        gr.update(visible=False, interactive=True),
        "Ready.",
        gr.update(visible=False),
        gr.update(value=""),
    )


def open_review_panel():
    return gr.update(visible=True)


def close_review_panel():
    return gr.update(visible=False), gr.update(value="")


async def run_workflow(
    stage: str,
    context_state: str,
    target_state: str,
    plan_state: str,
    instruction: str,
    pasted_text: str,
    resolution: str,
    cfg: float,
    steps: int,
    seed: int,
    total_generations: int,
    yolo: bool,
    progress=gr.Progress(),  # noqa: B008 - Gradio injects the progress tracker
):
    width, height = _parse_resolution(resolution)
    LOGGER.info(
        "Workflow request: stage=%s yolo=%s resolution=%sx%s steps=%s seed=%s",
        stage,
        yolo,
        width,
        height,
        steps,
        seed,
    )

    if yolo:
        context, target = _read_context(pasted_text)
        progress(0.05, desc="Planning")
        planned = await plan(
            context,
            target,
            model=LLM_MODEL,
            target_width=width,
            target_height=height,
        )
        progress(0.10, desc="Styling the approved structure")
        styled = await style(
            planned,
            methodology_text=context,
            figure_caption=target,
            model=LLM_MODEL,
        )
        progress(0.15, desc="Preparing diffusion prompt")
        prompt = await condense(
            styled,
            model=LLM_MODEL,
            target_width=width,
            target_height=height,
        )
        await _prepare_pipeline(progress, 0.18, 0.38)

        generations = min(4, max(1, int(total_generations)))
        generation_span = 0.60 / generations
        image_path = await _render_with_progress(
            progress,
            prompt,
            width,
            height,
            cfg,
            steps,
            seed,
            0.40,
            0.40 + generation_span,
            f"Generating diagram 1/{generations}",
        )

        for index in range(1, generations):
            segment_start = 0.40 + generation_span * index
            progress(
                segment_start,
                desc=f"Visual refine {index}/{generations - 1}",
            )
            prompt = await refine_prompt_from_image(
                prompt, image_path, target, planned, model=VISION_MODEL
            )
            image_path = await _render_with_progress(
                progress,
                prompt,
                width,
                height,
                cfg,
                steps,
                seed,
                segment_start,
                segment_start + generation_span,
                f"Generating diagram {index + 1}/{generations}",
            )
        progress(1.0, desc="Done")
        return (
            "done",
            context,
            target,
            planned,
            gr.update(value=prompt, label="Final prompt · editable"),
            image_path,
            gr.update(value="Run YOLO Again", interactive=True),
            gr.update(visible=True, interactive=True),
            f"YOLO complete: {generations} total generation(s), including {generations - 1} visual refinement(s).",
            gr.update(visible=False),
            gr.update(value=""),
        )

    if stage == "start":
        context, target = _read_context(pasted_text)
        progress(0.2, desc="Generating plan")
        planned = await plan(
            context,
            target,
            model=LLM_MODEL,
            target_width=width,
            target_height=height,
        )
        return (
            "plan",
            context,
            target,
            planned,
            gr.update(value=planned, label="Plan · review and edit"),
            None,
            gr.update(value="Approve Plan", interactive=True),
            gr.update(visible=False, interactive=True),
            "Plan ready. Review and edit it, then approve.",
            gr.update(visible=False),
            gr.update(value=""),
        )

    if stage == "plan":
        if not instruction.strip():
            raise gr.Error("The plan is empty.")
        progress(0.15, desc="Styling the approved structure")
        styled = await style(
            instruction,
            methodology_text=context_state,
            figure_caption=target_state,
            model=LLM_MODEL,
        )
        progress(0.3, desc="Preparing diffusion prompt")
        prompt = await condense(
            styled,
            model=LLM_MODEL,
            target_width=width,
            target_height=height,
        )
        return (
            "prompt",
            context_state,
            target_state,
            instruction,
            gr.update(value=prompt, label="Diffusion prompt · review and edit"),
            None,
            gr.update(value="Approve Prompt & Generate", interactive=True),
            gr.update(visible=False, interactive=True),
            "Prompt ready. Review and edit it, then approve generation.",
            gr.update(visible=False),
            gr.update(value=""),
        )

    if not instruction.strip():
        raise gr.Error("The diffusion prompt is empty.")
    await _prepare_pipeline(progress, 0.04, 0.38)
    image_path = await _render_with_progress(
        progress,
        instruction,
        width,
        height,
        cfg,
        steps,
        seed,
        0.40,
        1.0,
        "Generating diagram",
    )
    return (
        "done",
        context_state,
        target_state,
        plan_state,
        gr.update(value=instruction, label="Diffusion prompt · editable"),
        image_path,
        gr.update(value="Regenerate from Edited Prompt", interactive=True),
        gr.update(visible=True, interactive=True),
        "Diagram generated. Edit the prompt and regenerate if needed.",
        gr.update(visible=False),
        gr.update(value=""),
    )


async def review_generated_image(
    instruction: str,
    image_path: str | None,
    target_request: str,
    approved_plan: str,
    high_level_instruction: str,
    progress=gr.Progress(),  # noqa: B008 - Gradio injects the tracker
):
    if not image_path:
        raise gr.Error("Generate a diagram before asking for an image review.")
    if not instruction.strip():
        raise gr.Error("The diffusion prompt is empty.")
    progress(0.1, desc=f"{VISION_MODEL} is reviewing structure, text, arrows, and color")
    critique, revised_prompt = await analyze_and_refine_prompt_from_image(
        instruction,
        image_path,
        target_request,
        approved_plan,
        model=VISION_MODEL,
        high_level_instruction=high_level_instruction,
    )
    progress(1.0, desc="Review complete")
    return (
        gr.update(value=revised_prompt, label="AI-revised diffusion prompt · review and edit"),
        f"**{VISION_MODEL} image review**\n\n" + critique,
        gr.update(value="Generate Revised Diagram", interactive=True),
    )


def _session_key(request: gr.Request) -> str:
    session_hash = (request.session_hash or "").strip()
    if not session_hash:
        raise gr.Error("Browser session is unavailable. Refresh the page and try again.")
    return session_hash


def _normalise_job_token(value: str | None) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    try:
        return uuid.UUID(token).hex
    except (ValueError, AttributeError):
        return ""


def _cleanup_session_jobs_locked(now: float) -> None:
    expired = [
        key
        for key, record in _SESSION_JOBS.items()
        if record.get("state") != "running"
        and now - record.get("completed_at", record["created_at"])
        > _RESULT_TTL_SECONDS
    ]
    for key in expired:
        _SESSION_JOBS.pop(key, None)
    live_keys = set(_SESSION_JOBS)
    for session_key, job_key in list(_SESSION_CURRENT.items()):
        if job_key not in live_keys:
            _SESSION_CURRENT.pop(session_key, None)


class JobProgress:
    """Store progress text in memory; browser polling retrieves it independently."""

    def __init__(self, record: dict):
        self.record = record

    def __call__(self, _value=None, *, desc: str | None = None, **_kwargs):
        if desc and self.record.get("state") == "running":
            self.record["message"] = desc


async def _drive_session_job(job_key: str, record: dict, coroutine_factory) -> None:
    try:
        result = await coroutine_factory()
    except asyncio.CancelledError:
        LOGGER.info("Background job %s cancelled", job_key[:8])
        return
    except Exception as exc:
        LOGGER.exception("Background workflow failed")
        with _SESSION_JOBS_LOCK:
            if _SESSION_JOBS.get(job_key) is record:
                record["state"] = "error"
                record["error"] = str(exc).strip("'") or type(exc).__name__
                record["completed_at"] = time.monotonic()
        return
    with _SESSION_JOBS_LOCK:
        if _SESSION_JOBS.get(job_key) is record:
            record["state"] = "done"
            record["result"] = result
            record["completed_at"] = time.monotonic()
    elapsed = time.monotonic() - record["created_at"]
    LOGGER.info("Background job %s completed in %.1fs", job_key[:8], elapsed)


def _install_session_job(
    session_key: str,
    previous_token: str,
    record: dict,
    coroutine_factory,
) -> str:
    job_key = record["id"]
    with _SESSION_JOBS_LOCK:
        _cleanup_session_jobs_locked(time.monotonic())
        old_key = _normalise_job_token(previous_token) or _SESSION_CURRENT.get(
            session_key, ""
        )
        old = _SESSION_JOBS.get(old_key)
        if old and old.get("state") == "running":
            raise RuntimeError("A workflow is already running in this browser session.")
        if old_key:
            _SESSION_JOBS.pop(old_key, None)
        task = asyncio.create_task(
            _drive_session_job(job_key, record, coroutine_factory)
        )
        record["task"] = task
        _SESSION_JOBS[job_key] = record
        _SESSION_CURRENT[session_key] = job_key
    return job_key


async def submit_workflow_job(
    stage: str,
    context_state: str,
    target_state: str,
    plan_state: str,
    instruction: str,
    pasted_text: str,
    resolution: str,
    cfg: float,
    steps: int,
    seed: int,
    total_generations: int,
    yolo: bool,
    job_token: str,
    request: gr.Request,
):
    if (yolo or stage == "start") and not pasted_text.strip():
        return (
            job_token,
            gr.update(interactive=True),
            gr.update(interactive=True),
            "Error: paste the relevant paper context first.",
        )
    if stage in {"plan", "prompt", "done"} and not instruction.strip():
        return (
            job_token,
            gr.update(interactive=True),
            gr.update(interactive=True),
            "Error: the current Plan or Instruction is empty.",
        )

    session_key = _session_key(request)
    record = {
        "id": uuid.uuid4().hex,
        "state": "running",
        "message": "Starting…",
        "result": None,
        "error": None,
        "created_at": time.monotonic(),
    }
    progress = JobProgress(record)
    def run_job():
        return run_workflow(
            stage,
            context_state,
            target_state,
            plan_state,
            instruction,
            pasted_text,
            resolution,
            cfg,
            steps,
            seed,
            total_generations,
            yolo,
            progress=progress,
        )
    try:
        new_token = _install_session_job(
            session_key,
            job_token,
            record,
            run_job,
        )
    except RuntimeError as exc:
        return (
            job_token,
            gr.update(interactive=False),
            gr.update(interactive=False),
            str(exc),
        )
    return (
        new_token,
        gr.update(interactive=False),
        gr.update(interactive=False),
        "Starting background workflow…",
    )


async def submit_review_job(
    instruction: str,
    image_path: str | None,
    target_request: str,
    approved_plan: str,
    high_level_instruction: str,
    job_token: str,
    request: gr.Request,
):
    if not image_path or not instruction.strip():
        return (
            job_token,
            gr.update(interactive=True),
            gr.update(interactive=True),
            "Error: generate a diagram before requesting an image review.",
            gr.update(visible=True),
            gr.skip(),
        )
    session_key = _session_key(request)
    record = {
        "id": uuid.uuid4().hex,
        "state": "running",
        "message": f"Starting {VISION_MODEL} image review…",
        "result": None,
        "error": None,
        "created_at": time.monotonic(),
    }
    progress = JobProgress(record)

    async def run_review():
        revised_instruction, review_status, revised_run_button = (
            await review_generated_image(
                instruction,
                image_path,
                target_request,
                approved_plan,
                high_level_instruction,
                progress=progress,
            )
        )
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            revised_instruction,
            gr.skip(),
            revised_run_button,
            gr.update(visible=True, interactive=True),
            review_status,
            gr.update(visible=False),
            gr.update(value=""),
        )

    try:
        new_token = _install_session_job(
            session_key,
            job_token,
            record,
            run_review,
        )
    except RuntimeError as exc:
        return (
            job_token,
            gr.update(interactive=False),
            gr.update(interactive=False),
            str(exc),
            gr.update(visible=True),
            gr.skip(),
        )
    return (
        new_token,
        gr.update(interactive=False),
        gr.update(interactive=False),
        "Starting background image review…",
        gr.update(visible=False),
        gr.skip(),
    )


def poll_session_job(job_token: str, job_ack: str, request: gr.Request):
    session_key = _session_key(request)
    with _SESSION_JOBS_LOCK:
        _cleanup_session_jobs_locked(time.monotonic())
        job_key = _normalise_job_token(job_token) or _SESSION_CURRENT.get(
            session_key, ""
        )
        record = _SESSION_JOBS.get(job_key)
        if record is None:
            return tuple(gr.skip() for _ in range(12))
        state = record.get("state")
        if state == "running":
            message = record.get("message") or "Working…"
            elapsed = max(0, int(time.monotonic() - record["created_at"]))
            return (
                *tuple(gr.skip() for _ in range(6)),
                gr.update(interactive=False),
                gr.update(interactive=False),
                f"{message} · {elapsed}s",
                gr.update(visible=False),
                gr.skip(),
                gr.skip(),
            )
        if state == "reset":
            # Reset is intentionally sticky until the next submission replaces it.
            # This makes a later timer tick repair any stale pre-reset HTTP response.
            return (*copy.deepcopy(record["result"]), gr.skip())
        if _normalise_job_token(job_ack) == job_key:
            _SESSION_JOBS.pop(job_key, None)
            return tuple(gr.skip() for _ in range(12))

    if state == "done":
        return (*copy.deepcopy(record["result"]), gr.update(value=job_key))
    return (
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.update(interactive=True),
        gr.update(interactive=True),
        f"Error: {record.get('error') or 'background workflow failed.'}",
        gr.update(visible=False),
        gr.skip(),
        gr.update(value=job_key),
    )


def _cancel_task_threadsafe(task: asyncio.Task | None) -> None:
    """Cancel a session task on its owning event loop."""
    if task is None or task.done():
        return
    loop = task.get_loop()
    if loop.is_closed():
        return
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    if loop is current_loop:
        task.cancel()
    elif loop.is_running():
        loop.call_soon_threadsafe(task.cancel)
    else:
        task.cancel()


async def reset_session_job(job_token: str, _job_ack: str, request: gr.Request):
    session_key = _session_key(request)
    result = reset_workflow()
    with _SESSION_JOBS_LOCK:
        now = time.monotonic()
        _cleanup_session_jobs_locked(now)
        old_key = _normalise_job_token(job_token) or _SESSION_CURRENT.get(
            session_key, ""
        )
        old = _SESSION_JOBS.pop(old_key, None)
        if old:
            old["state"] = "cancelled"
        # Reuse the browser's known token. If the reset HTTP response is lost,
        # later polls can still retrieve the sticky reset result with that token.
        reset_token = old_key or uuid.uuid4().hex
        reset_record = {
            "id": reset_token,
            "state": "reset",
            "result": copy.deepcopy(result),
            "message": "Ready.",
            "created_at": now,
            "completed_at": now,
        }
        _SESSION_JOBS[reset_token] = reset_record
        _SESSION_CURRENT[session_key] = reset_token
    if old:
        cancel_event = old.get("cancel_event")
        if isinstance(cancel_event, threading.Event):
            cancel_event.set()
        _cancel_task_threadsafe(old.get("task"))
        # Let a same-loop task observe cancellation before returning to the browser.
        await asyncio.sleep(0)

    return (reset_token, "", *result)


def build_demo() -> gr.Blocks:
    if gr.__version__ != TESTED_GRADIO_VERSION:
        raise RuntimeError(
            f"This UI requires gradio=={TESTED_GRADIO_VERSION}; found {gr.__version__}. "
            "Install requirements-ui.txt."
        )

    with gr.Blocks(
        title="SciForma",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
        ),
        css=APP_CSS,
        analytics_enabled=False,
        fill_width=True,
        fill_height=True,
    ) as demo:
        stage = gr.State("start")
        context_state = gr.State("")
        target_state = gr.State("")
        plan_state = gr.State("")
        # Browser-held rather than server-session state, so a transient tunnel
        # reconnect cannot orphan a completed background result.
        job_token = gr.Textbox(value="", visible=False, container=False)
        job_ack = gr.Textbox(value="", visible=False, container=False)
        poll_timer = gr.Timer(value=1.0, active=True)

        with gr.Row(equal_height=False, elem_classes=["sf-main"]):
            with gr.Column(scale=3, min_width=0, elem_classes=["sf-left"]):
                with gr.Column(elem_classes=["sf-card", "sf-source"]):
                    gr.Markdown("### Input", elem_classes=["sf-title"])
                    pasted_text = gr.Textbox(
                        label="Paste your text",
                        placeholder="Paste the relevant paper context here...",
                        lines=18,
                        interactive=True,
                        elem_classes=["sf-editor", "sf-source-context"],
                    )

                with gr.Column(elem_classes=["sf-card", "sf-settings"]):
                    gr.Markdown("### Settings", elem_classes=["sf-title"])
                    gr.Markdown(
                        f"{LLM_MODEL} · SciForma-9B · FLUX.2-klein-base-9B",
                        elem_classes=["sf-note"],
                    )
                    resolution = gr.Dropdown(
                        label="Aspect ratio",
                        choices=ASPECT_CHOICES,
                        value=DEFAULT_RESOLUTION,
                    )
                    with gr.Row():
                        cfg = gr.Number(label="CFG", value=4.0)
                        steps = gr.Number(label="Steps", value=50, precision=0)
                        seed = gr.Number(label="Seed", value=42, precision=0)
                    with gr.Row():
                        total_generations = gr.Dropdown(
                            choices=[1, 2, 3, 4],
                            value=2,
                            label="Total generations",
                            info="YOLO only. 1 means no visual refinement.",
                        )
                        yolo = gr.Checkbox(
                            label="Automatic refinement (YOLO)",
                            value=False,
                        )

            with gr.Column(scale=7, min_width=0, elem_classes=["sf-right"]):
                with gr.Column(elem_classes=["sf-card", "sf-draft"]):
                    gr.Markdown("### Draft", elem_classes=["sf-title"])
                    instruction = gr.Textbox(
                        label="Instruction",
                        placeholder="The current Plan or diffusion prompt appears here.",
                        lines=4,
                        show_copy_button=True,
                        elem_classes=["sf-editor", "sf-draft-editor"],
                    )
                    with gr.Row():
                        run_button = gr.Button(
                            "Start", variant="primary", elem_classes=["sf-button"]
                        )
                        review_button = gr.Button(
                            "AI Review Image",
                            variant="primary",
                            visible=False,
                            elem_classes=["sf-button"],
                        )
                        reset_button = gr.Button(
                            "Reset", variant="primary", elem_classes=["sf-button"]
                        )
                    with gr.Column(
                        visible=False,
                        elem_classes=["sf-review-panel"],
                    ) as review_panel:
                        gr.Markdown(
                            "**How should AI revise this image?**",
                            elem_classes=["sf-review-heading"],
                        )
                        review_instruction = gr.Textbox(
                            label="High-level instruction",
                            placeholder=(
                                "For example: simplify the composition, replace text-heavy "
                                "boxes with clear icons, and make the outer margins uniform."
                            ),
                            lines=3,
                            elem_classes=["sf-editor"],
                        )
                        with gr.Row():
                            cancel_review_button = gr.Button("Cancel")
                            confirm_review_button = gr.Button(
                                "Analyze & Revise",
                                variant="primary",
                                elem_classes=["sf-button"],
                            )
                    status = gr.Markdown("Ready.", elem_classes=["sf-status"])

                with gr.Column(elem_classes=["sf-card", "sf-preview-card"]):
                    gr.Markdown("### Preview", elem_classes=["sf-title"])
                    result = gr.Image(
                        label="Generated diagram",
                        type="filepath",
                        height="100%",
                        interactive=False,
                        sources=[],
                        show_label=False,
                        elem_classes=["sf-preview"],
                    )

        reset_button.click(
            reset_session_job,
            inputs=[job_token, job_ack],
            outputs=[
                job_token,
                job_ack,
                stage,
                context_state,
                target_state,
                plan_state,
                instruction,
                result,
                run_button,
                review_button,
                status,
                review_panel,
                review_instruction,
            ],
            queue=False,
            show_progress="hidden",
            api_name="reset",
            show_api=False,
        )
        run_button.click(
            submit_workflow_job,
            [
                stage,
                context_state,
                target_state,
                plan_state,
                instruction,
                pasted_text,
                resolution,
                cfg,
                steps,
                seed,
                total_generations,
                yolo,
                job_token,
            ],
            [job_token, run_button, review_button, status],
            api_name="submit_workflow",
            show_api=False,
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )
        review_button.click(
            open_review_panel,
            outputs=review_panel,
            api_name="open_review",
            show_api=False,
            queue=False,
            show_progress="hidden",
        )
        cancel_review_button.click(
            close_review_panel,
            outputs=[review_panel, review_instruction],
            api_name="close_review",
            show_api=False,
            queue=False,
            show_progress="hidden",
        )
        confirm_review_button.click(
            submit_review_job,
            [
                instruction,
                result,
                target_state,
                plan_state,
                review_instruction,
                job_token,
            ],
            [
                job_token,
                run_button,
                review_button,
                status,
                review_panel,
                review_instruction,
            ],
            api_name="submit_review",
            show_api=False,
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )
        poll_timer.tick(
            poll_session_job,
            inputs=[job_token, job_ack],
            outputs=[
                stage,
                context_state,
                target_state,
                plan_state,
                instruction,
                result,
                run_button,
                review_button,
                status,
                review_panel,
                review_instruction,
                job_ack,
            ],
            api_name="poll",
            show_api=False,
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )
    return demo


def preload_local_pipeline() -> None:
    """Validate and load the prepared cache before accepting browser requests."""
    LOGGER.info("Validating the prepared local model cache")
    try:
        _download_model_files(local_files_only=True)
    except Exception as exc:
        raise SystemExit(
            "Local model cache is incomplete. Run "
            "`python generate/prepare_ui_models.py` first. "
            "HF_TOKEN is required only by that preparation command."
        ) from exc
    LOGGER.info("Loading the prepared local model into GPU memory")
    _load_pipeline()
    LOGGER.info("Local model is ready")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if os.environ.get("SCIFORMA_UI_PRELOAD", "1") != "0":
        preload_local_pipeline()
    build_demo().launch(
        server_name=os.environ.get("SCIFORMA_UI_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("SCIFORMA_UI_PORT", "7860")),
        show_error=True,
        show_api=False,
        ssr_mode=False,
        pwa=False,
    )
