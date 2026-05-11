# ==============================================================================
# Aircraft Engine Prognostics - Development Environment
# ==============================================================================
# RTX 5070 (Blackwell sm_120) 지원을 위해 CUDA 12.8 기반
# ==============================================================================

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

LABEL maintainer="이강태 <gangtae9050@gmail.com>"
LABEL description="Aircraft turbofan engine RUL prediction development environment"

ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim htop unzip tree \
    python3.10 python3-pip python3.10-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ---- PyTorch with CUDA 12.8 (RTX 5070 support) ----
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ---- Project dependencies ----
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Jupyter Lab ----
RUN pip install --no-cache-dir \
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    kaggle>=1.5.0

# ---- Environment variables ----
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8888 6006

CMD ["/bin/bash"]
