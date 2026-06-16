import random
from pathlib import Path
from tqdm import tqdm
from statistics import mean

from babygrad.data import load_csv, prepare_supervised_data
from babygrad.tensor import Tensor
from babygrad.metrics import accuracy
from babygrad.nn import CCE, SGD, Linear, ReLU, Sequential, Softmax
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
            Linear(128, splits.y_train.ncol),
            Softmax(),
        ],
    )

    epochs = 30
    batch_size = 10
    optimizer = SGD(model.parameters(), 0.1)
    criterion = CCE()

    progress_epoch = tqdm(range(epochs), desc="train (epochs)")
    for e in progress_epoch:
        recorder.set_step(e)

        batch_idx = list(range(0, splits.x_train.nrow))
        random.shuffle(batch_idx)
        random_x_data = [splits.x_train[i : i + 1] for i in batch_idx]
        flat_x = []
        for x in random_x_data:
            flat_x.extend(x.data)
        random_x_train = Tensor(flat_x, (len(random_x_data), splits.x_train.ncol))

        random_y_data = [splits.y_train[i : i + 1] for i in batch_idx]
        flat_y = []
        for y in random_y_data:
            flat_y.extend(y.data)
        random_y_train = Tensor(flat_y, (len(random_y_data), splits.y_train.ncol))

        progress_batch = tqdm(
            range(0, random_x_train.nrow, batch_size),
            desc="train (batch)",
            leave=False,
            position=1,
        )

        accum_loss = []
        accum_acc = []
        loss: Tensor | None = None
        for b in progress_batch:
            optimizer.zero_grad()
            x = random_x_train[b : b + batch_size]
            y = random_y_train[b : b + batch_size]

            y_pred = model.forward(x)
            loss = criterion.forward(y, y_pred)
            acc = accuracy(y, y_pred)
            accum_loss.append(loss.data[0])
            accum_acc.append(acc)

            progress_batch.set_postfix(loss=f"{loss.data[0]:.4f}", acc=f"{acc:.3f}")
            loss.backward()
            optimizer.step()

        # validation
        y_pred = model.forward(splits.x_val)
        validation_loss = criterion.forward(splits.y_val, y_pred)

        if loss is not None:
            recorder.capture(root=loss)
        recorder.record("loss", mean(accum_loss))
        recorder.record("acc", mean(accum_acc))
        recorder.record("val_loss", validation_loss.data[0])

    y_pred = model.forward(splits.x_test)
    loss = criterion.forward(splits.y_test, y_pred)
    acc = accuracy(splits.y_test, y_pred)
    print(f"test loss: {loss.data[0]:.4f}, test acc: {acc:.3f}")

    # visualizer = GraphVisualizer(loss)
    # visualizer.draw_architecture(save_path="./architecture.svg")
    # visualizer.draw_combined(save_path="./combined.svg")
    # visualizer.draw_computation(save_path="./computation.svg")

    visualizer = PlotVisualizer(recorder.history)
    visualizer.plot_scalar(["loss", "val_loss"])
    visualizer.plot_scalar(["acc"])
    visualizer.plot_ridge("Linear_0/weights")
    visualizer.plot_ridge("Linear_0/weights/grad", clip_quantiles=(0.01, 0.99))
    visualizer.plot_ridge("Linear_2/weights/grad", clip_quantiles=(0.01, 0.99))


if __name__ == "__main__":
    train_iris()
