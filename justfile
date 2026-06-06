train:
    python main.py

lint:
    uv run ruff check babygrad tests

typecheck:
    uv run pyright

test:
    uv run pytest
