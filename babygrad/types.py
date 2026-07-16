"""Shared type vocabulary: structural aliases and domain value types."""

from enum import Enum, auto

type Number = int | float
type Shape = tuple[int, ...]

type Tag = str
type Step = int
type History = dict[Tag, dict[Step, HistoryValue]]
type HistoryValue = float | list[float]


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
