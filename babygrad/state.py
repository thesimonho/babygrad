"""
ContextVars so we don't need to prop drill model state through the entire graph.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_scope: ContextVar[str | None] = ContextVar("scope", default=None)
_is_training: ContextVar[bool] = ContextVar("is_training", default=True)


@contextmanager
def bound(var: ContextVar, value):
    """
    Set a ContextVar to a value for the duration of the block, then restore the
    previous value on exit (even if the block raises).
    """
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)
