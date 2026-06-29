# Contributing

## Setup

```bash
git clone https://github.com/parthj732005/llm4teach-reflection
cd llm4teach-reflection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install pytest numpy
```

## Running Tests

```bash
pytest tests/ -v
```

Tests require only `pytest` and `numpy` — no GPU, no Ollama, no MiniGrid needed.

## Submitting a PR

1. Fork the repo and create a branch
2. Make your changes without breaking existing tests
3. Run `pytest tests/` and confirm all tests pass
4. Open a pull request with a clear description of what changed and why
