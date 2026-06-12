from pathlib import Path
from tqdm import tqdm
from babygrad.data import load_csv, prepare_supervised_data
from babygrad.nn import CCE, SGD, ReLU, Sequential, Linear, Softmax
from babygrad.metrics import accuracy

from babygrad.plot import Visualizer
from babygrad.observer import Observer


def train_iris():
    dataset = load_csv(Path("./data/iris.csv"))
    dataset.one_hot = True
    dataset.target_col_idx = 4
    splits = prepare_supervised_data(dataset)

    observer = Observer()
    model = Sequential(
        [
            Linear(splits.x_train.ncol, 128),
            ReLU(),
            Linear(128, splits.y_train.ncol),
            Softmax(),
        ],
    )

    epochs = 30
    optimizer = SGD(model.parameters(), 0.1)

    progress = tqdm(range(epochs), desc="training")
    for i in progress:
        observer.set_step(i)
        optimizer.zero_grad()
        y_pred = model.forward(splits.x_train, observer)
        loss = CCE(splits.y_train, y_pred)
        acc = accuracy(splits.y_train, y_pred)
        observer.record("loss", loss.data[0])
        observer.record("acc", acc)

        progress.set_postfix(loss=f"{loss.data[0]:.4f}", acc=f"{acc:.3f}")
        loss.backward()
        model.report_grads(observer)
        optimizer.step()

    visualizer = Visualizer()
    visualizer.plot_scalar("loss", observer.history)
    visualizer.plot_scalar("acc", observer.history)
    visualizer.plot_ridge("Linear_0/weights", observer.history)
    visualizer.plot_ridge(
        "Linear_0/grad", observer.history, clip_quantiles=(0.01, 0.99)
    )
    visualizer.plot_ridge(
        "Linear_2/grad", observer.history, clip_quantiles=(0.01, 0.99)
    )


if __name__ == "__main__":
    train_iris()
