@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM LLM4Teach — Windows quick-setup (no Docker needed)
REM Run this once on a new machine, then use run_viz.bat (live dashboard),
REM run_train.bat (CLI), or open Research_Ablation_Notebook.ipynb.
REM ─────────────────────────────────────────────────────────────────────────────
setlocal
pushd "%~dp0"

echo.
echo ============================================================
echo  LLM4Teach Setup (Windows, CPU-only)
echo ============================================================
echo.

REM 1. Create a virtual environment
echo [1/5] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM 2. Upgrade pip
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip

REM 3. Install CPU-only PyTorch
echo [3/5] Installing PyTorch (CPU only)...
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

REM 4. Install remaining requirements (core + notebook + live web dashboard)
echo [4/5] Installing remaining requirements...
pip install gymnasium==1.3.0 minigrid==3.1.0 numpy==1.26.4 ^
    tensorboard==2.20.0 opencv-python==4.9.0.80 ^
    matplotlib==3.10.0 requests==2.31.0 ^
    jupyter==1.0.0 ipykernel==6.29.0 ^
    fastapi==0.110.0 "uvicorn[standard]==0.29.0"

REM 5. Remind user to install + set up Ollama
echo [5/5] Ollama setup (OPTIONAL — offline mode works with no LLM)...
echo.
echo  *** OPTIONAL — only needed for the ONLINE planner/reflector ***
echo  Download and install Ollama from: https://ollama.ai/download/windows
echo  Then open a NEW terminal and run:
echo.
echo     ollama serve
echo     ollama pull qwen2.5:3b        (planner + reflector — REQUIRED for online)
echo     ollama pull qwen2.5:7b        (OPTIONAL — richer reflections, heavier)
echo.
echo ============================================================
echo  Setup complete! Choose how to run:
echo.
echo     run_train.bat                    Command-line training (main.py)
echo     jupyter notebook                 Open LLM4Teach_Kaggle_Research.ipynb (3-phase study)
echo ============================================================
echo.
popd
endlocal
pause
