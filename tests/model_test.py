import pytest

from babygrad.data import DataSplit, Sample
from babygrad.nn.losses import MSE
from babygrad.nn.model import Model, TrainConfig, Trainer
from babygrad.nn.modules import BatchNorm, Linear, Residual, Sequential
from babygrad.nn.optimizers import SGD
from babygrad.nn.schedulers import ConstantLR
from babygrad.observers import Recorder
from babygrad.tensor import Tensor
from babygrad.types import NodeKind


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

    # the fit path also captures the last batch's params/grads through the reporter's
    # per-batch tracer -> end_epoch capture; guard that wiring, not just record()
    assert "Linear_0/weights/grad" in recorder.history

    # accumulators reset each epoch: the final epoch's recorded mean is its lone
    # batch's loss, i.e. the returned tensor. Lifetime drift would break this.
    last_epoch = epochs - 1
    assert recorder.history["loss"][last_epoch] == pytest.approx(final_loss.data[0])
