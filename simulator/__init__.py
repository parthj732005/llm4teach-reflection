"""
simulator/ — Logging backbone for LLM4Teach.

Active module
-------------
  trajectory_logger.py   The episode/step logger used by Game.collect(). Writes
                         training.log, trajectories.jsonl, prompts.log and the
                         per-episode metrics.csv (mirrored to results/<phase>/
                         episodes.csv by experiment_runner.run_phase).

      from simulator.trajectory_logger import TrajectoryLogger, EpisodeRecord, StepRecord

Live visualization now lives in the `viz/` package (a FastAPI + WebSocket web
dashboard) — launch it with `run_viz.bat` (or `python viz/run_viz.py`).

Legacy (matplotlib / streamlit) helpers — live_dashboard.py, visualize_training.py,
render_episode.py — are kept only for the standalone `main.py` CLI and are
superseded by `viz/` for interactive use.
"""
