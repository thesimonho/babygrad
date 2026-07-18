import sys
from pathlib import Path
from typing import TYPE_CHECKING

from babygrad.data import CSVDataset, DataLoader, split_train_val_test
from babygrad.metrics import Accuracy
from babygrad.nn.activations import ReLU, Softmax
from babygrad.nn.losses import CCE
from babygrad.nn.model import Model, TrainConfig, Trainer
from babygrad.nn.modules import (
    BatchNorm,
    Dropout,
    Linear,
    Residual,
    Sequential,
)
from babygrad.nn.optimizers import SGD, Adam
from babygrad.nn.schedulers import ConstantLR, CosineAnnealingLR
from babygrad.observers import Recorder, Tracer
from babygrad.tracing import tracing
from babygrad.viz.attribution import attribute
from babygrad.viz.graph import GraphVisualizer
from babygrad.viz.plot import PlotVisualizer

if TYPE_CHECKING:
    from babygrad.nn.modules import Module


def train_iris():
    dataset = CSVDataset(Path("./data/iris.csv"), 4, True)
    train, val, test = split_train_val_test(dataset, one_hot=True)

    root = Sequential(
        [
            Linear(train.n_features, 128),
            ReLU(),
            Residual(
                Sequential(
                    [Linear(128, 128), BatchNorm(128), ReLU(), Dropout(0.8)],
                ),
            ),
            Linear(128, train.n_targets),
            Softmax(),
        ],
    )

    config = TrainConfig(
        epochs=30,
        batch_size=10,
        optimizer=SGD(root.parameters()),
        scheduler=ConstantLR(0.1),
        criterion=CCE(0.1, collapse=True),
        metrics=[Accuracy()],
    )
    model = Model(root)
    recorder = Recorder()
    trainer = Trainer(model, config, recorder)
    train_loss = trainer.fit(train, val)
    trainer.test(test)

    if train_loss is not None:
        # Trace a fresh forward: the fit() loss carries no scope attribution, so
        # re-run one batch under a tracer to feed the graph views.
        demo_x, demo_y = DataLoader(train, batch_size=config.batch_size).full_batch()
        tracer = Tracer()
        with tracing(tracer):
            loss = config.criterion(demo_y, model.forward(demo_x))
        visualizer = GraphVisualizer(loss, attribute(tracer.records))
        visualizer.draw_architecture(save_path="./plots/architecture.svg")
        visualizer.draw_computation(save_path="./plots/computation.svg")
        visualizer.draw_combined(save_path="./plots/combined.svg")

    plotter = PlotVisualizer(recorder.history)
    plotter.plot_scalar(["loss", "val_loss"], save_path="./plots/loss.png")
    plotter.plot_scalar(["Accuracy"], save_path="./plots/accuracy.png")
    plotter.plot_ridge("Linear_0/weights", save_path="./plots/weights.png")
    plotter.plot_ridge(
        "Linear_3/weights/grad",
        clip_quantiles=(0.01, 0.99),
        save_path="./plots/grad.png",
    )


def train_resnet():
    dataset = CSVDataset(Path("./data/concentric_circles.csv"), target_col_idx=2)
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

    config = TrainConfig(
        epochs=20,
        batch_size=64,
        optimizer=Adam(root.parameters()),
        scheduler=CosineAnnealingLR(T_0=1, T_mult=2),
        criterion=CCE(collapse=True),
        metrics=[Accuracy()],
    )
    model = Model(root)
    trainer = Trainer(model, config)
    trainer.fit(train, val)
    trainer.test(test)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "resnet":
        train_resnet()
    else:
        train_iris()
