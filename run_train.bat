@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM LLM4Teach — Start training from the CLI (Windows)
REM   Run setup.bat once first. Online LLM is optional: if Ollama isn't running,
REM   the planner/reflector automatically fall back to offline (pre-computed plans).
REM   NOTE: most research runs use Research_Ablation_Notebook.ipynb (3-phase
REM   pipeline) or the live dashboard (run_viz.bat); this CLI path uses main.py.
REM ─────────────────────────────────────────────────────────────────────────────
setlocal
pushd "%~dp0"

if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"

echo Starting training with Qwen planner + reflection...
echo (For online LLM: run "ollama serve" and "ollama pull qwen2.5:3b" in another terminal)
echo.

REM Reflector reuses the planner model (qwen2.5:3b) so it works with a single
REM pulled model. Forcing an un-pulled model (e.g. qwen2.5:7b) makes every
REM reflector call 404 → 0 reflections. Pull 7b first if you want richer summaries.
python main.py train ^
    --task simpledoorkey ^
    --savedir run1 ^
    --llm_backend ollama ^
    --llm_model qwen2.5:3b ^
    --reflection ^
    --reflection_backend ollama ^
    --reflection_model qwen2.5:3b

popd
endlocal
pause
