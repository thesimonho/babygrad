"""Observation seam: lets a tracer bracket module/loss forward passes without
the ML classes knowing what is watching. With no active tracer, ``__call__`` is
a plain passthrough, so the graphing concern stays opt-in and out of the core.
"""

from __future__ import annotations

from abc import ABC
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol

from babygrad.state import bound

if TYPE_CHECKING:
    from collections.abc import Callable

    from babygrad.tensor import Tensor


class TracerLike(Protocol):
    """What ``Traceable.__call__`` needs from a tracer: to be told when a
    module's forward is entered and when it exits."""

    def enter(self, module: Traceable, inputs: tuple[Tensor, ...]) -> None: ...
    def exit(self, module: Traceable, output: Tensor) -> None: ...


class NullTracer:
    """The default tracer: sees nothing, does nothing. Keeps a bracketed call a
    zero-cost passthrough whenever no one is tracing."""

    def enter(self, module: Traceable, inputs: tuple[Tensor, ...]) -> None:
        pass

    def exit(self, module: Traceable, output: Tensor) -> None:
        pass


_active_tracer: ContextVar[TracerLike] = ContextVar(
    "active_tracer", default=NullTracer()
)


def tracing(tracer: TracerLike):
    """Make ``tracer`` the active tracer for the duration of the block."""
    return bound(_active_tracer, tracer)


class Traceable(ABC):
    """Base for anything a tracer can bracket (Modules, Losses). ``__call__``
    wraps ``forward`` so the active tracer sees the module boundary, leaving the
    subclass's ``forward`` free of any graphing concern.

    ``forward`` is declared as a permissive callable rather than a fixed-arity
    abstract method so subclasses can keep their own signatures — ``Module`` takes
    one input, ``Loss`` takes two. Each enforces its own abstractness.

    ``collapse`` is a display hint (draw this whole scope as one box) that both
    Modules and Losses carry; the tracer reads it onto the scope."""

    collapse: bool
    forward: Callable[..., Tensor]

    def __call__(self, *inputs: Tensor) -> Tensor:
        tracer = _active_tracer.get()
        tracer.enter(self, inputs)
        output = self.forward(*inputs)
        tracer.exit(self, output)
        return output
