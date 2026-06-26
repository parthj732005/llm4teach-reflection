#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LLM4Teach — Linux/Mac quick-setup (no Docker needed)
# Run once on a new machine, then: bash run_train.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "============================================================"
echo " LLM4Teach Setup (Linux/Mac, CPU-only)"
echo "============================================================"
echo ""

# 1. Create virtual environment
echo "[1/5] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# 3. CPU-only PyTorch
echo "[3/5] Installing PyTorch (CPU only)..."
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Remaining requirements
echo "[4/5] Installing remaining requirements..."
pip install \
    gymnasium==1.3.0 \
    minigrid==3.1.0 \
    numpy==1.26.4 \
    tensorboard==2.20.0 \
    opencv-python-headless==4.9.0.80 \
    matplotlib==3.10.0 \
    streamlit==1.35.0 \
    requests==2.31.0 \
    jupyter==1.0.0 \
    ipykernel==6.29.0

# 5. Install Ollama
echo "[5/5] Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

echo ""
echo "Pulling Qwen models (this may take a few minutes)..."
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b

echo ""
echo "============================================================"
echo " Setup complete! To start training, run:"
echo ""
echo "    bash run_train.sh"
echo "============================================================"
echo ""
