lint:
    uv run ruff check babygrad tests

typecheck:
    uv run pyright

test:
    uv run pytest

notebooks:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# start the standalone dashboard server; leave it running, then stream a run to it
dashboard:
    uv run python -m babygrad.viz.serve

# train a model and stream it live to a running dashboard server
train model="iris":
    uv run python main.py {{ model }} --dashboard
