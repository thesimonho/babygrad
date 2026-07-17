from babygrad.nn.model import Model
from babygrad.nn.modules import BatchNorm, Linear, Residual, Sequential
from babygrad.tensor import Tensor
from babygrad.types import NodeKind


def test_residual_add_scoped_to_residual_not_inner_block():
    """A Residual runs its own ``input + output`` add *after* its nested block
    returns, with no scope push of its own. That add must therefore be scoped to
    the enclosing Residual, not left carrying the inner block's scope.

    This is the discriminating case for the scope context manager: restoring to
    the *prior* scope (via the reset token) yields the Residual's name, whereas
    resetting to the container's own name would wrongly leave the inner
    Sequential's scope active when the add runs.
    """
    inner = Sequential([Linear(2, 2)])
    residual = Residual(inner)
    model = Model(Sequential([residual]))

    # the Residual's add is the graph's final tensor for this model
    output = model.forward(Tensor([1.0, 2.0], shape=(1, 2), kind=NodeKind.VIEW))

    assert residual.name == "Sequential_0/Residual_0"
    assert output.scope == residual.name


def test_batchnorm_reads_training_mode_even_when_nested_in_residual():
    """``is_training`` must reach a BatchNorm buried inside a Residual. The old
    per-layer threading dropped the flag at the Residual boundary (Residual does
    not forward it to its block); the ambient ContextVar reaches it regardless
    of nesting depth.

    Running stats update in training but must stay frozen under eval. Asserting
    the update/no-update split on ``running_mean`` proves the flag actually
    arrived with the right value at both settings.
    """
    batch_norm = BatchNorm(2)
    model = Model(Sequential([Residual(Sequential([batch_norm]))]))
    x = Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2), kind=NodeKind.VIEW)

    frozen_running_mean = list(batch_norm.running_mean)

    model.eval(x)
    # eval never touches running stats -> proves is_training arrived as False
    assert batch_norm.running_mean == frozen_running_mean

    model.forward(x)
    # training updates them -> proves the same ambient path carries True
    assert batch_norm.running_mean != frozen_running_mean
