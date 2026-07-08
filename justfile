train model="":
    uv run python main.py {{ model }}

lint:
    uv run ruff check babygrad tests

typecheck:
    uv run pyright

test:
    uv run pytest

notebooks:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
