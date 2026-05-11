# ==============================================================================
# Aircraft Engine Prognostics - Development Environment
# ==============================================================================
# Base: PyTorch 2.0.1 with CUDA 11.7 (GPU support included; falls back to CPU)
# ==============================================================================

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="이강태 <gangtae9050@gmail.com>"
LABEL description="Aircraft turbofan engine RUL prediction development environment"

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ---- Python environment ----
WORKDIR /workspace

# 의존성 먼저 설치 (Docker layer caching 최적화)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Jupyter Lab 추가
RUN pip install --no-cache-dir \
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    kaggle>=1.5.0

# ---- Environment variables ----
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---- Expose ports ----
EXPOSE 8888   
EXPOSE 6006   

# ---- Default command ----
CMD ["/bin/bash"]