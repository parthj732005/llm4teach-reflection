#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LLM4Teach — Start training (Linux/Mac)
# ─────────────────────────────────────────────────────────────────────────────

source venv/bin/activate

# Start Ollama in background if not already running
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server in background..."
    ollama serve &
    sleep 3
fi

echo "Starting training with Qwen planner + reflection..."
echo ""

python main.py train \
    --task simpledoorkey \
    --savedir run1 \
    --llm_backend ollama \
    --llm_model qwen2.5:3b \
    --reflection \
    --reflection_backend ollama \
    --reflection_model qwen2.5:3b
