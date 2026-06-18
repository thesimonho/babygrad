from pathlib import Path
from tqdm import tqdm
from statistics import mean
from math import ceil

from babygrad.data import load_csv, prepare_supervised_data, create_minibatches
from babygrad.tensor import Tensor
from babygrad.metrics import accuracy
from babygrad.nn import CCE, SGD, Linear, ReLU, Sequential, Softmax, BatchNorm
from babygrad.recorder import Recorder
from babygrad.viz.plot import PlotVisualizer


def train_iris():
    dataset = load_csv(Path("./data/iris.csv"))
    dataset.one_hot = True
    dataset.target_col_idx = 4
    splits = prepare_supervised_data(dataset)

    recorder = Recorder()
    model = Sequential(
        [
            Linear(splits.x_train.ncol, 128),
            ReLU(),
            BatchNorm(128),
            Linear(128, splits.y_train.ncol),
            Softmax(),
        ],
    )

    epochs = 30
    batch_size = 10
    optimizer = SGD(model.parameters(), 0.1)
    criterion = CCE()

    n_batches = ceil(splits.x_train.nrow / batch_size)
    progress_epoch = tqdm(range(epochs), desc="train (epochs)")
    progress_batch = tqdm(
        total=n_batches,
        desc="train (batch)",
        leave=False,
        position=1,
    )

    for e in progress_epoch:
        recorder.set_step(e)

        minibatches = create_minibatches(splits.x_train, splits.y_train, batch_size)
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
        y_pred = model.eval(splits.x_val)
        validation_loss = criterion.forward(splits.y_val, y_pred)

        if loss is not None:
            recorder.capture(root=loss)
        recorder.record("loss", mean(accum_loss))
        recorder.record("acc", mean(accum_acc))
        recorder.record("val_loss", validation_loss.data[0])

    progress_batch.close()

    y_pred = model.eval(splits.x_test)
    loss = criterion.forward(splits.y_test, y_pred)
    acc = accuracy(splits.y_test, y_pred)
    print(f"\ntest loss: {loss.data[0]:.4f}, test acc: {acc:.3f}")

    # visualizer = GraphVisualizer(loss)
    # visualizer.draw_architecture(save_path="./architecture.svg")
    # visualizer.draw_combined(save_path="./combined.svg")
    # visualizer.draw_computation(save_path="./computation.svg")

    visualizer = PlotVisualizer(recorder.history)
    visualizer.plot_scalar(["loss", "val_loss"])
    visualizer.plot_scalar(["acc"])
    visualizer.plot_ridge("Linear_0/weights")
    visualizer.plot_ridge("Linear_3/weights/grad", clip_quantiles=(0.01, 0.99))


if __name__ == "__main__":
    train_iris()
