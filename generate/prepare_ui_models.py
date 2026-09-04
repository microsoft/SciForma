#!/usr/bin/env python3
"""Pre-download and validate the pinned model files used by the local UI.

This command deliberately does not import ``latex_to_diagram_ui``.  It reads
the UI's literal model pins from source so that preparing a deployment does
not import Gradio, Diffusers, Torch, or the LLM client.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UI_SOURCE = Path(__file__).with_name("latex_to_diagram_ui.py")

# These manifests describe the exact files at the revisions pinned by the UI.
# A pin change intentionally requires updating this manifest as well.
EXPECTED_PINS = {
    "LoYuXrqw/SciForma-9B": "70cc9b0665681ec63a02f3067253481c9dc75184",
    "black-forest-labs/FLUX.2-klein-base-9B": "32773329fbe7e81a90ef971740e8ba4b0364ecf3",
}

SCIFORMA_FILES = {
    "transformer/config.json": 532,
    "transformer/diffusion_pytorch_model.safetensors.index.json": 24_874,
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors": 3_828_363_264,
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors": 3_791_662_800,
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors": 3_925_877_600,
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors": 3_925_877_632,
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors": 2_685_409_392,
}

BASE_FILES = {
    "model_index.json": 422,
    "scheduler/scheduler_config.json": 486,
    "text_encoder/config.json": 1_538,
    "text_encoder/generation_config.json": 214,
    "text_encoder/model.safetensors.index.json": 32_914,
    "text_encoder/model-00001-of-00004.safetensors": 4_902_257_696,
    "text_encoder/model-00002-of-00004.safetensors": 4_915_960_368,
    "text_encoder/model-00003-of-00004.safetensors": 4_983_068_496,
    "text_encoder/model-00004-of-00004.safetensors": 1_580_230_264,
    "tokenizer/added_tokens.json": 707,
    "tokenizer/chat_template.jinja": 4_168,
    "tokenizer/merges.txt": 1_671_853,
    "tokenizer/special_tokens_map.json": 613,
    "tokenizer/tokenizer.json": 11_422_654,
    "tokenizer/tokenizer_config.json": 5_404,
    "tokenizer/vocab.json": 2_776_833,
    "vae/config.json": 821,
    "vae/diffusion_pytorch_model.safetensors": 168_120_878,
}

BASE_EXCLUDED_FILE = "flux-2-klein-base-9b.safetensors"
BASE_EXCLUDED_SIZE = 18_157_185_168

JSON_FILES = {
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "vae/config.json",
}


@dataclass(frozen=True)
class IndexManifest:
    path: str
    shards: frozenset[str]
    total_size: int


@dataclass(frozen=True)
class ModelManifest:
    label: str
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    files: dict[str, int]
    indexes: tuple[IndexManifest, ...]
    excluded_file: str | None = None
    excluded_size: int = 0


@dataclass
class CheckReport:
    manifest: ModelManifest
    snapshot: Path
    missing: list[str]
    invalid: list[str]
    present_bytes: int
    offline_resolved: bool

    @property
    def required_bytes(self) -> int:
        return sum(self.manifest.files.values())

    @property
    def ready(self) -> bool:
        return not self.missing and not self.invalid and self.offline_resolved


def _literal_assignments(path: Path, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = ast.literal_eval(value_node)
    missing = sorted(names - values.keys())
    if missing:
        raise RuntimeError(f"Missing literal UI model constants: {', '.join(missing)}")
    return values


def _load_ui_pins() -> tuple[str, str, str, str]:
    values = _literal_assignments(
        UI_SOURCE,
        {
            "PUBLIC_MODEL",
            "PUBLIC_BASE_MODEL",
            "MODEL_REVISIONS",
            "BASE_MODEL_REVISIONS",
        },
    )
    model_id = values["PUBLIC_MODEL"]
    base_id = values["PUBLIC_BASE_MODEL"]
    model_revision = values["MODEL_REVISIONS"].get(model_id)
    base_revision = values["BASE_MODEL_REVISIONS"].get(base_id)
    pins = {model_id: model_revision, base_id: base_revision}
    if pins != EXPECTED_PINS:
        raise RuntimeError(
            "The UI model pins changed. Update EXPECTED_PINS and the exact file manifests "
            f"in {Path(__file__).name}. UI pins: {pins!r}"
        )
    return model_id, model_revision, base_id, base_revision


def _manifests() -> tuple[ModelManifest, ModelManifest]:
    model_id, model_revision, base_id, base_revision = _load_ui_pins()
    sciforma_shards = frozenset(
        Path(path).name for path in SCIFORMA_FILES if path.endswith(".safetensors")
    )
    base_shards = frozenset(
        Path(path).name
        for path in BASE_FILES
        if path.startswith("text_encoder/model-") and path.endswith(".safetensors")
    )
    return (
        ModelManifest(
            label="SciForma transformer",
            repo_id=model_id,
            revision=model_revision,
            allow_patterns=("transformer/*",),
            files=SCIFORMA_FILES,
            indexes=(
                IndexManifest(
                    path="transformer/diffusion_pytorch_model.safetensors.index.json",
                    shards=sciforma_shards,
                    total_size=18_157_162_496,
                ),
            ),
        ),
        ModelManifest(
            label="FLUX base components",
            repo_id=base_id,
            revision=base_revision,
            allow_patterns=(
                "model_index.json",
                "scheduler/*",
                "text_encoder/*",
                "tokenizer/*",
                "vae/*",
            ),
            files=BASE_FILES,
            indexes=(
                IndexManifest(
                    path="text_encoder/model.safetensors.index.json",
                    shards=base_shards,
                    total_size=16_381_470_720,
                ),
            ),
            excluded_file=BASE_EXCLUDED_FILE,
            excluded_size=BASE_EXCLUDED_SIZE,
        ),
    )


def _cache_root() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE).expanduser().resolve()


def _snapshot_path(cache_root: Path, manifest: ModelManifest) -> Path:
    repo_folder = "models--" + manifest.repo_id.replace("/", "--")
    return cache_root / repo_folder / "snapshots" / manifest.revision


def _gb(size: int) -> str:
    return f"{size / 1_000_000_000:.2f} GB"


def _validate_index(snapshot: Path, expected: IndexManifest) -> list[str]:
    path = snapshot / expected.path
    if not path.is_file():
        return []  # Already reported as a missing manifest file.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{expected.path}: invalid JSON ({exc})"]

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return [f"{expected.path}: missing or empty weight_map"]
    referenced = frozenset(weight_map.values())
    errors: list[str] = []
    if referenced != expected.shards:
        missing = sorted(expected.shards - referenced)
        extra = sorted(referenced - expected.shards)
        errors.append(
            f"{expected.path}: shard map mismatch; missing={missing or 'none'}, "
            f"unexpected={extra or 'none'}"
        )
    total_size = payload.get("metadata", {}).get("total_size")
    if total_size != expected.total_size:
        errors.append(
            f"{expected.path}: metadata.total_size expected {expected.total_size}, "
            f"found {total_size!r}"
        )
    return errors


def _offline_resolves(manifest: ModelManifest) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            manifest.repo_id,
            revision=manifest.revision,
            allow_patterns=list(manifest.allow_patterns),
            local_files_only=True,
            token=False,
        )
    except Exception:  # noqa: BLE001 - the detailed missing report is printed separately
        return False
    return True


def _check(manifest: ModelManifest, cache_root: Path) -> CheckReport:
    snapshot = _snapshot_path(cache_root, manifest)
    missing: list[str] = []
    invalid: list[str] = []
    present_bytes = 0

    if not snapshot.is_dir():
        missing.extend(manifest.files)
    else:
        for relative, expected_size in manifest.files.items():
            path = snapshot / relative
            if not path.is_file():
                missing.append(relative)
                continue
            try:
                found_size = path.stat().st_size
            except OSError as exc:
                invalid.append(f"{relative}: cannot stat ({exc})")
                continue
            present_bytes += found_size
            if found_size != expected_size:
                invalid.append(
                    f"{relative}: expected {_gb(expected_size)} ({expected_size} bytes), "
                    f"found {_gb(found_size)} ({found_size} bytes)"
                )
                continue
            if relative in JSON_FILES:
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    invalid.append(f"{relative}: invalid JSON ({exc})")

        for index in manifest.indexes:
            invalid.extend(_validate_index(snapshot, index))

    invalid = list(dict.fromkeys(invalid))

    return CheckReport(
        manifest=manifest,
        snapshot=snapshot,
        missing=missing,
        invalid=invalid,
        present_bytes=present_bytes,
        offline_resolved=_offline_resolves(manifest),
    )


def _print_report(report: CheckReport) -> None:
    manifest = report.manifest
    state = "READY" if report.ready else "INCOMPLETE"
    print(f"\n[{state}] {manifest.label}")
    print(f"  repository: {manifest.repo_id}")
    print(f"  revision:   {manifest.revision}")
    print(f"  snapshot:   {report.snapshot}")
    print(
        f"  required:   {len(manifest.files)} files, {_gb(report.required_bytes)}; "
        f"present {_gb(report.present_bytes)}"
    )
    print(
        f"  offline:    {'OK (local_files_only, no token)' if report.offline_resolved else 'FAILED'}"
    )

    if report.missing:
        print("  missing:")
        for relative in report.missing:
            expected = manifest.files.get(relative)
            suffix = f" ({_gb(expected)})" if expected is not None else ""
            print(f"    - {relative}{suffix}")
    else:
        print("  missing:    none")

    if report.invalid:
        print("  invalid:")
        for message in report.invalid:
            print(f"    - {message}")
    else:
        print("  invalid:    none")

    if manifest.excluded_file:
        excluded_path = report.snapshot / manifest.excluded_file
        if excluded_path.is_file():
            actual = excluded_path.stat().st_size
            cached = f"already cached ({_gb(actual)}); left untouched"
        else:
            cached = "not cached"
        print(
            f"  excluded:   {manifest.excluded_file} "
            f"({_gb(manifest.excluded_size)} monolithic transformer; {cached})"
        )


def _download(manifest: ModelManifest, max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    print(f"\n[DOWNLOAD] {manifest.label}")
    print(f"  repository: {manifest.repo_id}@{manifest.revision}")
    print(f"  include:    {', '.join(manifest.allow_patterns)}")
    if manifest.excluded_file:
        print(f"  exclude:    {manifest.excluded_file} ({_gb(manifest.excluded_size)})")
    snapshot_download(
        manifest.repo_id,
        revision=manifest.revision,
        allow_patterns=list(manifest.allow_patterns),
        ignore_patterns=[manifest.excluded_file] if manifest.excluded_file else None,
        token=token,
        max_workers=max_workers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download and validate the exact SciForma/FLUX files needed by "
            "generate/latex_to_diagram_ui.py."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Do not use the network; only validate the pinned snapshots already in cache.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel Hugging Face downloads (default: 8).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_workers < 1:
        print("error: --max-workers must be at least 1", file=sys.stderr)
        return 2

    try:
        manifests = _manifests()
        cache_root = _cache_root()
    except Exception as exc:  # noqa: BLE001 - CLI should provide a concise deployment error
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mode = "offline check" if args.check_only else "download and check"
    print("SciForma UI model preparation")
    print(f"  mode:       {mode}")
    print(f"  cache root: {cache_root}")
    if not args.check_only:
        print(f"  HF_TOKEN:   {'set' if os.environ.get('HF_TOKEN') else 'not set'}")

    download_errors: list[str] = []
    if not args.check_only:
        for manifest in manifests:
            try:
                _download(manifest, args.max_workers)
            except Exception as exc:  # noqa: BLE001 - continue to print exact missing files
                message = f"{manifest.repo_id}: {type(exc).__name__}: {exc}"
                download_errors.append(message)
                print(f"  download failed: {message}", file=sys.stderr)

    try:
        reports = [_check(manifest, cache_root) for manifest in manifests]
    except Exception as exc:  # noqa: BLE001 - keep deployment failures concise
        print(f"error: model cache validation failed: {exc}", file=sys.stderr)
        return 2
    for report in reports:
        _print_report(report)

    required = sum(report.required_bytes for report in reports)
    present = sum(
        min(report.present_bytes, report.required_bytes) for report in reports
    )
    print(f"\nRequired deployment payload: {_gb(required)}; present: {_gb(present)}")

    if download_errors:
        print("Download errors:", file=sys.stderr)
        for message in download_errors:
            print(f"  - {message}", file=sys.stderr)
        if any(
            report.manifest.repo_id == "black-forest-labs/FLUX.2-klein-base-9B"
            and not report.ready
            for report in reports
        ):
            print(
                "Accept the FLUX.2-klein-base-9B license, set HF_TOKEN, and rerun this command.",
                file=sys.stderr,
            )

    if download_errors or not all(report.ready for report in reports):
        print("\nModel preparation FAILED.")
        return 1

    print(
        "\nModel preparation complete. The UI can now load both snapshots offline without HF_TOKEN."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
