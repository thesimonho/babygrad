from pathlib import Path
from tqdm import tqdm
from babygrad.data import load_csv, prepare_supervised_data
from babygrad.nn import CCE, SGD, ReLU, Sequential, Linear, Softmax
from babygrad.metrics import accuracy

from babygrad.recorder import Recorder


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
    optimizer = SGD(model.parameters(), 0.1)
    criterion = CCE()

    progress = tqdm(range(epochs), desc="training")
    for i in progress:
        recorder.set_step(i)
        optimizer.zero_grad()
        y_pred = model.forward(splits.x_train)
        loss = criterion.forward(splits.y_train, y_pred)
        acc = accuracy(splits.y_train, y_pred)
        recorder.record("loss", loss.data[0])
        recorder.record("acc", acc)

        progress.set_postfix(loss=f"{loss.data[0]:.4f}", acc=f"{acc:.3f}")
        loss.backward()
        recorder.capture(loss)
        optimizer.step()

    y_pred = model.forward(splits.x_train)
    loss = criterion.forward(splits.y_train, y_pred)

    # visualizer = GraphVisualizer(loss)
    # visualizer.draw_architecture(save_path="./architecture.svg")
    # visualizer.draw_combined(save_path="./combined.svg")
    # visualizer.draw_computation(save_path="./computation.svg")

    # visualizer = PlotVisualizer(recorder.history)
    # visualizer.plot_scalar("loss", recorder.history)
    # visualizer.plot_scalar("acc", recorder.history)
    # visualizer.plot_ridge("Linear_0/weights", recorder.history)
    # visualizer.plot_ridge(
    #     "Linear_0/weights/grad", recorder.history, clip_quantiles=(0.01, 0.99)
    # )
    # visualizer.plot_ridge(
    #     "Linear_2/weights/grad", recorder.history, clip_quantiles=(0.01, 0.99)
    # )


if __name__ == "__main__":
    train_iris()
