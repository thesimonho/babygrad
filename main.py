import sys
from math import ceil
from pathlib import Path
from random import seed
from statistics import mean
from time import perf_counter

from tqdm import tqdm

from babygrad.data import CSVDataset, DataLoader, split_train_val_test
from babygrad.metrics import accuracy
from babygrad.nn.activations import ReLU, Softmax
from babygrad.nn.losses import CCE
from babygrad.nn.modules import (
    BatchNorm,
    Dropout,
    Linear,
    Model,
    Module,
    Residual,
    Sequential,
)
from babygrad.nn.optimizers import SGD, Adam
from babygrad.nn.schedulers import ConstantLR, CosineAnnealingLR
from babygrad.recorder import Recorder
from babygrad.state import _is_training, bound
from babygrad.tensor import Tensor
from babygrad.viz.graph import GraphVisualizer


def train_iris():
    dataset = CSVDataset(Path("./data/iris.csv"), 4, True)
    train, val, test = split_train_val_test(dataset, one_hot=True)

    recorder = Recorder()
    root = Sequential(
        [
            Linear(train.n_features, 128),
            ReLU(),
            Residual(
                Sequential(
                    [Linear(128, 128), BatchNorm(128), ReLU(), Dropout(0.8)],
                )
            ),
            Linear(128, train.n_targets),
            Softmax(),
        ],
    )
    model = Model(root)

    epochs = 30
    batch_size = 10
    optimizer = SGD(root.parameters())
    scheduler = ConstantLR(0.1)
    criterion = CCE(0.1)

    n_batches = ceil(train.nrow / batch_size)
    progress_epoch = tqdm(range(epochs), desc="train (epochs)")
    progress_batch = tqdm(
        total=n_batches,
        desc="train (batch)",
        leave=False,
        position=1,
    )

    train_loss: Tensor | None = None
    for e in progress_epoch:
        optimizer.lr = scheduler(e)
        recorder.step = e

        minibatches = DataLoader(train, batch_size)
        progress_batch.reset()

        accum_loss = []
        accum_acc = []
        for x, y in minibatches:
            optimizer.zero_grad()

            y_pred = model.forward(x)
            with bound(_is_training, True):
                train_loss = criterion.forward(y, y_pred)
            acc = accuracy(y, y_pred)
            accum_loss.append(train_loss.data[0])
            accum_acc.append(acc)

            train_loss.backward()
            optimizer.step()

            progress_batch.set_postfix(
                loss=f"{train_loss.data[0]:.4f}", acc=f"{acc:.3f}"
            )
            progress_batch.update()

        progress_batch.refresh()

        # validation
        val_loader = DataLoader(val)
        x, y = val_loader.full_batch()
        y_pred = model.eval(x)
        with bound(_is_training, False):
            validation_loss = criterion.forward(y, y_pred)

        if train_loss is not None:
            recorder.capture(root=train_loss)
        recorder.record("loss", mean(accum_loss))
        recorder.record("acc", mean(accum_acc))
        recorder.record("val_loss", validation_loss.data[0])

    progress_batch.close()

    test_loader = DataLoader(test)
    x, y = test_loader.full_batch()
    y_pred = model.eval(x)
    with bound(_is_training, False):
        test_loss = criterion.forward(y, y_pred)

    acc = accuracy(y, y_pred)
    print(f"\ntest loss: {test_loss.data[0]:.4f}, test acc: {acc:.3f}")

    if train_loss is not None:
        visualizer = GraphVisualizer(train_loss)
        visualizer.draw_architecture(save_path="./plots/architecture.svg")
        visualizer.draw_computation(save_path="./plots/computation.svg")
        visualizer.draw_combined(save_path="./plots/combined.svg")

    # with PlotVisualizer(recorder.history) as visualizer:
    #     visualizer.plot_scalar(["loss", "val_loss"])
    #     visualizer.plot_scalar(["acc"])
    #     visualizer.plot_ridge("Linear_0/weights")
    #     visualizer.plot_ridge("Linear_3/weights/grad", clip_quantiles=(0.01, 0.99))


def train_resnet():
    seed(0)  # deterministic split, shuffling, and weight init

    dataset = CSVDataset(Path("./data/concentric_circles.csv"), target_col_idx=2)
    dataset.data = dataset.data[:1250]  # same 1250-row slice as the notebook
    train, val, test = split_train_val_test(dataset, one_hot=True)

    width = 16  # hidden width
    blocks = 18  # 18 blocks * 2 Linears = 36 hidden layers

    def block_body():
        return [
            BatchNorm(width),
            ReLU(),
            Linear(width, width),
            BatchNorm(width),
            ReLU(),
            Linear(width, width),
        ]

    layers: list[Module] = [Linear(train.n_features, width)]
    for _ in range(blocks):
        layers.append(Residual(Sequential(block_body())))
    layers += [Linear(width, train.n_targets), Softmax()]
    root = Sequential(layers)

    model = Model(root)

    epochs = 40
    batch_size = 64
    optimizer = Adam(model.root.parameters())
    scheduler = CosineAnnealingLR(T_0=1, T_mult=2)
    criterion = CCE()
    val_x, val_y = DataLoader(val).full_batch()

    n_batches = ceil(train.nrow / batch_size)
    for e in range(epochs):
        epoch_start = perf_counter()
        optimizer.lr = scheduler(e)

        batch_losses = []
        batches = tqdm(
            DataLoader(train, batch_size),
            total=n_batches,
            desc=f"epoch {e + 1}/{epochs}",
            leave=False,
        )
        for x, y in batches:
            optimizer.zero_grad()
            loss = criterion.forward(y, model.forward(x))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.data[0])

        val_pred = model.eval(val_x)
        val_acc = accuracy(val_y, val_pred)

        with bound(_is_training, False):
            val_loss = criterion.forward(val_y, val_pred)

        epoch_time = perf_counter() - epoch_start
        print(
            f"epoch {e + 1}/{epochs}  "
            f"train_loss {mean(batch_losses):.4f}  "
            f"val_loss {val_loss.data[0]:.4f}  "
            f"val_acc {val_acc:.3f}  "
            f"({epoch_time:.2f}s)"
        )

    test_x, test_y = DataLoader(test).full_batch()
    test_pred = model.eval(test_x)
    print(
        f"\ntest loss: {criterion.forward(test_y, test_pred).data[0]:.4f}, "
        f"test acc: {accuracy(test_y, test_pred):.3f}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "resnet":
        train_resnet()
    else:
        train_iris()
