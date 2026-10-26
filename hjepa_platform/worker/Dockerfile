# GPU worker image for H-JEPA Studio: upstream H-JEPA (pinned) + the platform agent.
# CUDA 12.8 wheels of torch 2.7.1 support Blackwell (RTX 5090, sm_120) as well as
# Ada/Ampere cards (RTX 4090/3090, A5000, L4).
#
#   docker build -t ghcr.io/<you>/hjepa-worker:latest hjepa_platform/worker
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HJEPA_SRC=/opt/H-JEPA \
    HJEPA_HOME=/workspace/hjepa_home \
    HJW_WORKDIR=/workspace/hjw \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl zstd ffmpeg \
        libegl1 libgl1 libgles2 libglib2.0-0 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
RUN uv python install 3.11.13 && uv venv /opt/venv --python 3.11.13
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv

ARG HJEPA_REPO=https://github.com/kevinghst/H-JEPA.git
ARG HJEPA_COMMIT=97054ecaff83e13839195ec4b92d74c08a0127c5
RUN git clone "$HJEPA_REPO" /opt/H-JEPA \
    && git -C /opt/H-JEPA checkout "$HJEPA_COMMIT"
RUN cd /opt/H-JEPA \
    && uv pip install -e ".[train,env]" \
        --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    && uv pip install "huggingface_hub[hf_transfer]>=0.30" pyyaml pillow

COPY hjepa_worker /opt/hjepa_worker/hjepa_worker
ENV PYTHONPATH=/opt/hjepa_worker
WORKDIR /workspace

# HJW_MODE=job (default) runs one job and exits; HJW_MODE=serve hosts a sim session.
CMD ["python", "-m", "hjepa_worker.agent"]
