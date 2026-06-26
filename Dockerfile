# ─────────────────────────────────────────────────────────────────────────────
# LLM4Teach — CPU-only Docker image
#
# Build:  docker build -t llm4teach .
# Run:    docker compose up        (see docker-compose.yml)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System packages ───────────────────────────────────────────────────────────
# libgl1 + libglib2.0 needed by opencv-python headless
# curl needed to install Ollama inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Install Ollama (runs inside the container, CPU mode) ──────────────────────
RUN curl -fsSL https://ollama.ai/install.sh | sh

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the pip install
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than CUDA build)
RUN pip install --no-cache-dir \
        torch==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir \
        gymnasium==1.3.0 \
        minigrid==3.1.0 \
        numpy==1.26.4 \
        tensorboard==2.20.0 \
        "opencv-python-headless==4.9.0.80" \
        matplotlib==3.10.0 \
        streamlit==1.35.0 \
        requests==2.31.0 \
        jupyter==1.0.0 \
        ipykernel==6.29.0

# ── Copy the project ──────────────────────────────────────────────────────────
COPY . .

# ── Pull Qwen models at build time (baked into the image) ─────────────────────
# Comment these out if you want a smaller image and prefer to pull at runtime.
# They are large: qwen2.5:3b ≈ 2 GB, qwen2.5:7b ≈ 4.7 GB
RUN ollama serve & sleep 5 && \
    ollama pull qwen2.5:3b && \
    ollama pull qwen2.5:7b && \
    pkill ollama || true

# ── Expose ports ──────────────────────────────────────────────────────────────
# 11434 = Ollama API
# 6006  = TensorBoard
# 8501  = Streamlit dashboard
# 8888  = Jupyter
EXPOSE 11434 6006 8501 8888

# ── Entrypoint ────────────────────────────────────────────────────────────────
# start.sh starts Ollama in background then runs whatever CMD is given
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
CMD ["python", "main.py", "train", \
     "--task", "SimpleDoorKey", \
     "--savedir", "run1", \
     "--llm_backend", "ollama", \
     "--llm_model", "qwen2.5:3b", \
     "--reflection", \
     "--reflection_backend", "ollama", \
     "--reflection_model", "qwen2.5:7b"]
