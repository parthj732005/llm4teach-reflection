#!/bin/bash
# docker/start.sh
# Starts Ollama in the background, waits for it to be ready, then runs the user command.

set -e

echo "==> Starting Ollama server (CPU mode)..."
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is accepting connections (up to 30s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "==> Ollama is ready."
        break
    fi
    echo "    Waiting for Ollama... ($i/30)"
    sleep 1
done

echo "==> Running: $@"
exec "$@"
