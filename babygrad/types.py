"""Shared type vocabulary: structural aliases and domain value types."""

from dataclasses import dataclass
from enum import Enum, auto

type Number = int | float
type Shape = tuple[int, ...]

type Tag = str
type Step = int
type History = dict[Tag, dict[Step, HistoryValue]]
type HistoryValue = float | list[float]


def series_tag(label: str, role: str) -> Tag:
    """The canonical tag for a recorded value series: ``"<module label>/<role>"``.

    The single source of this format so the recorder (which writes the history)
    and the graph JSON (which references a tag to link a node to its history)
    cannot drift out of sync.
    """
    return f"{label}/{role}"


def grad_tag(base: Tag) -> Tag:
    """The tag for a value series' gradient companion: ``"<base>/grad"``."""
    return f"{base}/grad"


class NodeKind(Enum):
    """The role a graph node plays, stamped at creation by whoever knows it.

    Every node (tensor or op) answers `.kind`, so consumers dispatch on the
    role directly instead of inferring it from which attributes are set.

    Stamped at the authoritative site for each role:
      - VIEW:           structural Tensor view (e.g. slice of a batch)
      - INPUT / TARGET: where data enters the graph (Sequential.forward / Loss)
      - PARAMETER:      Linear.__init__ (model state)
      - OP:             the Op class itself (every op is an OP)
      - OP_RESULT:      Op.forward (the tensor an op produces)
      - LAYER_OUTPUT:   Sequential.forward (a named layer boundary)
      - LOSS:           the Loss base (the scalar the whole graph hinges on)
      - CONSTANT:       Constant tensors (e.g. BatchNorm running stats)

    OP_RESULT is the default for any computed tensor; LAYER_OUTPUT and LOSS
    are more specific roles that override it (most-specific owner stamps last).
    """

    VIEW = auto()
    INPUT = auto()
    TARGET = auto()
    PARAMETER = auto()
    OP = auto()
    OP_RESULT = auto()
    LAYER_OUTPUT = auto()
    LOSS = auto()
    CONSTANT = auto()


@dataclass
class Scope:
    """One node in the module containment tree: a box that holds graph nodes and
    nests inside its ``outer_scope``.

    Structure is declared, not parsed: ``outer_scope`` names the enclosing scope
    directly, so consumers walk pointers instead of splitting the ``id`` path.
    """

    id: str
    label: str
    outer_scope: str | None
    collapsed: bool
