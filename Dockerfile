# SciForma — Inference Docker Image
# Usage:
#   docker build -t sciforma-infer .
#   docker run --gpus all \
#       -e HF_TOKEN=hf_xxx \
#       -e AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com \
#       -v $(pwd)/outputs:/workspace/outputs \
#       sciforma-infer \
#       python generate/latex_to_diagram.py --latex /workspace/paper.tex --caption "..."

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="SciForma"
LABEL org.opencontainers.image.description="Structure-Faithful Scientific Diagram Generation"
LABEL org.opencontainers.image.source="https://github.com/microsoft/SciForma"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/SciForma

WORKDIR /workspace/SciForma

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git curl \
    && ln -sf python3.10 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# diffusers from git (Flux2Klein support)
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers.git@3996788b"

# Project dependencies (inference only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY sciforma/      ./sciforma/
COPY generate/      ./generate/
COPY eval/          ./eval/
COPY .env.example   ./.env.example

CMD ["python", "generate/latex_to_diagram.py", "--help"]
