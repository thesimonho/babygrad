from math import ceil
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from babygrad.data import CSVDataset, DataLoader, split_train_val_test
from babygrad.metrics import accuracy
from babygrad.nn import CCE, SGD, BatchNorm, Linear, ReLU, Sequential, Residual, Softmax
from babygrad.recorder import Recorder
from babygrad.tensor import Tensor
from babygrad.viz.graph import GraphVisualizer


def train_iris():
    dataset = CSVDataset(Path("./data/iris.csv"), 4, True)
    train, val, test = split_train_val_test(dataset, one_hot=True)

    recorder = Recorder()
    model = Sequential(
        [
            Linear(train.n_features, 128),
            ReLU(),
            Residual(
                Sequential(
                    [
                        Linear(128, 128),
                        ReLU(),
                        BatchNorm(128),
                    ],
                )
            ),
            Linear(128, train.n_targets),
            Softmax(),
        ],
    )

    epochs = 30
    batch_size = 10
    optimizer = SGD(model.parameters(), 0.1)
    criterion = CCE()

    n_batches = ceil(train.nrow / batch_size)
    progress_epoch = tqdm(range(epochs), desc="train (epochs)")
    progress_batch = tqdm(
        total=n_batches,
        desc="train (batch)",
        leave=False,
        position=1,
    )

    for e in progress_epoch:
        recorder.set_step(e)

        minibatches = DataLoader(train, batch_size)
        progress_batch.reset()

        accum_loss = []
        accum_acc = []
        loss: Tensor | None = None
        for x, y in minibatches:
            optimizer.zero_grad()

            y_pred = model.forward(x)
            loss = criterion.forward(y, y_pred)
            acc = accuracy(y, y_pred)
            accum_loss.append(loss.data[0])
            accum_acc.append(acc)

            loss.backward()
            optimizer.step()

            progress_batch.set_postfix(loss=f"{loss.data[0]:.4f}", acc=f"{acc:.3f}")
            progress_batch.update()

        progress_batch.refresh()

        # validation
        val_loader = DataLoader(val)
        x, y = val_loader.full_batch()
        y_pred = model.eval(x)
        validation_loss = criterion.forward(y, y_pred)

        if loss is not None:
            recorder.capture(root=loss)
        recorder.record("loss", mean(accum_loss))
        recorder.record("acc", mean(accum_acc))
        recorder.record("val_loss", validation_loss.data[0])

    progress_batch.close()

    test_loader = DataLoader(test)
    x, y = test_loader.full_batch()
    y_pred = model.eval(x)
    loss = criterion.forward(y, y_pred)
    acc = accuracy(y, y_pred)
    print(f"\ntest loss: {loss.data[0]:.4f}, test acc: {acc:.3f}")

    visualizer = GraphVisualizer(loss)
    visualizer.draw_architecture(save_path="./architecture.svg")
    visualizer.draw_combined(save_path="./combined.svg")
    visualizer.draw_computation(save_path="./computation.svg")

    # with PlotVisualizer(recorder.history) as visualizer:
    #     visualizer.plot_scalar(["loss", "val_loss"])
    #     visualizer.plot_scalar(["acc"])
    #     visualizer.plot_ridge("Linear_0/weights")
    #     visualizer.plot_ridge("Linear_3/weights/grad", clip_quantiles=(0.01, 0.99))


if __name__ == "__main__":
    train_iris()
