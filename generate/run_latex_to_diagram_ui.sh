#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
cd "$repo_dir"

python - <<'PY'
import os
import socket

host = os.environ.get("SCIFORMA_UI_HOST", "127.0.0.1")
port = int(os.environ.get("SCIFORMA_UI_PORT", "7860"))
probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.settimeout(0.5)
    if probe.connect_ex((probe_host, port)) == 0:
        raise SystemExit(
            f"Port {port} is already in use. Open the existing UI or set "
            "SCIFORMA_UI_PORT to another port."
        )
PY

python - <<'PY'
import importlib
import hashlib
import sys
from pathlib import Path

expected = {
    "accelerate": "1.12.0",
    "diffusers": "0.37.0.dev0",
    "gradio": "5.50.0",
    "huggingface_hub": "0.36.2",
    "openai": "2.14.0",
    "PIL": "11.3.0",
    "safetensors": "0.7.0",
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "transformers": "4.57.3",
    "azure.identity": "1.25.1",
}
errors = []
if sys.version.split()[0] != "3.10.19":
    errors.append(f"Python: expected 3.10.19, found {sys.version.split()[0]}")
for package, wanted in expected.items():
    module = importlib.import_module(package)
    found = getattr(module, "__version__", "unknown")
    if found != wanted:
        errors.append(f"{package}: expected {wanted}, found {found}")

import diffusers

diffusers_root = Path(diffusers.__file__).parent
expected_diffusers_hashes = {
    "pipelines/flux2/pipeline_flux2_klein.py": "e625460449da51819ba940db2967306a554728b057275719fa4e6d67592f44c0",
    "models/transformers/transformer_flux2.py": "0107b8d4c127d8753e2c1b9ebf3b275c88d042a796c483d5c7965924c16a9eef",
}
for relative, wanted in expected_diffusers_hashes.items():
    path = diffusers_root / relative
    found = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    if found != wanted:
        errors.append(f"diffusers/{relative}: installed source does not match the validated pin")
if errors:
    raise SystemExit("Validated environment mismatch:\n  " + "\n  ".join(errors))
print("Validated SciForma UI environment.")
PY

python generate/prepare_ui_models.py --check-only

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python generate/latex_to_diagram_ui.py
