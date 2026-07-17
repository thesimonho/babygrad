import pytest

from babygrad.data import DataSplit, Sample
from babygrad.nn.losses import MSE
from babygrad.nn.model import Model, TrainConfig, Trainer
from babygrad.nn.modules import BatchNorm, Linear, Residual, Sequential
from babygrad.nn.optimizers import SGD
from babygrad.nn.schedulers import ConstantLR
from babygrad.recorder import Recorder
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


def test_stamp_materialises_scope_tree_with_outer_links():
    """stamp_name_and_scope builds an explicit Scope per module, linked to its
    enclosing scope, so consumers read structure without splitting the id path."""
    model = Model(Sequential([Residual(Sequential([Linear(2, 2)]))]))

    top = model.scopes["Sequential_0"]
    assert top.outer_scope is None
    assert top.label == "Sequential_0"

    residual = model.scopes["Sequential_0/Residual_0"]
    assert residual.outer_scope == "Sequential_0"
    assert residual.label == "Residual_0"


def test_stamp_records_collapse_flag_on_scope():
    """A module's collapse flag lands on its Scope, matching the legacy string set."""
    model = Model(Sequential([Sequential([Linear(2, 2)], collapse=True)]))

    assert model.scopes["Sequential_0/Sequential_0"].collapsed is True
    assert model.scopes["Sequential_0"].collapsed is False
    # additive slice: the legacy string set still agrees
    assert "Sequential_0/Sequential_0" in model.collapsed_scopes


def _tiny_regression_split() -> DataSplit:
    """Three-row, two-feature regression split — enough to run real batches
    cheaply. No one-hot mapping, so ``n_targets`` is just the scalar target.
    """
    return DataSplit(
        rows=[
            Sample([0.0, 0.0], [0.0]),
            Sample([1.0, 0.0], [1.0]),
            Sample([0.0, 1.0], [1.0]),
        ]
    )


def test_fit_records_one_epoch_mean_per_epoch_without_lifetime_drift():
    """The reporter must reset its per-epoch accumulators in ``start_epoch``;
    if it does not, each recorded epoch loss becomes a running mean over every
    batch seen so far rather than that epoch's own mean.

    Making ``batch_size`` cover the whole split forces exactly one batch per
    epoch, so a correctly-reset reporter records a loss equal to that single
    batch's loss — which is the very tensor ``fit`` returns from the last epoch.
    Under the drift bug the final record would instead average across all epochs
    and, because SGD moves the weights between epochs, diverge from it. That gap
    is what the last assertion detects.
    """
    train = _tiny_regression_split()
    val = _tiny_regression_split()
    epochs = 3

    root = Sequential([Linear(2, 1)])
    config = TrainConfig(
        epochs=epochs,
        batch_size=train.nrow,  # one batch per epoch -> epoch mean == that batch's loss
        optimizer=SGD(root.parameters()),
        scheduler=ConstantLR(0.1),
        criterion=MSE(),
    )
    recorder = Recorder()
    final_loss = Trainer(Model(root), config, recorder).fit(train, val)

    # fit ran and propagated the last batch's loss tensor back out
    assert final_loss is not None

    # exactly one recorded point per epoch, no gaps or duplicates
    assert len(recorder.history["loss"]) == epochs
    assert len(recorder.history["val_loss"]) == epochs

    # accumulators reset each epoch: the final epoch's recorded mean is its lone
    # batch's loss, i.e. the returned tensor. Lifetime drift would break this.
    last_epoch = epochs - 1
    assert recorder.history["loss"][last_epoch] == pytest.approx(final_loss.data[0])
