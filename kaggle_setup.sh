#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LLM4Teach — Kaggle Setup Script
#
# Paste each CELL block into a separate Kaggle notebook code cell, in order.
# Enable "Internet" in Kaggle Settings before running.
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════
# CELL 1 — Unzip project + install Python libraries
# ══════════════════════════════════════════════════════════════════

cd /kaggle/working

# Unzip your uploaded project (adjust filename if different)
unzip -q /kaggle/input/llm4teach/LLM4Teach-main.zip -d .

# Install CPU-only PyTorch (Kaggle already has torch but this pins the version)
pip install -q torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -q \
    gymnasium==1.3.0 \
    minigrid==3.1.0 \
    numpy==1.26.4 \
    tensorboard==2.20.0 \
    opencv-python-headless==4.9.0.80 \
    matplotlib==3.10.0 \
    streamlit==1.35.0 \
    requests==2.31.0

echo "✅ Python libraries installed."


# ══════════════════════════════════════════════════════════════════
# CELL 2 — Install Ollama
# ══════════════════════════════════════════════════════════════════

curl -fsSL https://ollama.ai/install.sh | sh
echo "✅ Ollama installed."


# ══════════════════════════════════════════════════════════════════
# CELL 3 — Start Ollama server in background
# ══════════════════════════════════════════════════════════════════

ollama serve &> /tmp/ollama.log &
sleep 5  # wait for server to boot

# Confirm it's alive
curl -sf http://localhost:11434/api/tags && echo "✅ Ollama server is running." || echo "❌ Ollama failed to start — check /tmp/ollama.log"


# ══════════════════════════════════════════════════════════════════
# CELL 4 — Pull Qwen models (takes ~10 min, run once per session)
# ══════════════════════════════════════════════════════════════════

echo "Pulling qwen2.5:3b (planner, ~2 GB)..."
ollama pull qwen2.5:3b

echo "Pulling qwen2.5:7b (reflector, ~4.7 GB)..."
ollama pull qwen2.5:7b

echo "✅ Both models ready."
ollama list


# ══════════════════════════════════════════════════════════════════
# CELL 5 — Run training
# ══════════════════════════════════════════════════════════════════

cd /kaggle/working/LLM4Teach-main

python main.py train \
    --task SimpleDoorKey \
    --savedir run1 \
    --llm_backend ollama \
    --llm_model qwen2.5:3b \
    --reflection \
    --reflection_backend ollama \
    --reflection_model qwen2.5:7b

# ══════════════════════════════════════════════════════════════════
# CELL 6 (optional) — Save outputs so they survive session end
# ══════════════════════════════════════════════════════════════════

cp -r /kaggle/working/LLM4Teach-main/log /kaggle/working/output_logs
cp -r /kaggle/working/LLM4Teach-main/saved_model /kaggle/working/output_models 2>/dev/null || true

echo "✅ Logs and models copied to /kaggle/working/ (will be saved automatically)"
