from pathlib import Path
from tqdm import tqdm
from babygrad.data import load_csv, prepare_supervised_data
from babygrad.nn import CCE, SGD, ReLU, Sequential, Linear, Softmax
from babygrad.metrics import accuracy


def train_iris():
    dataset = load_csv(Path("./data/iris.csv"))
    dataset.one_hot = True
    dataset.target_col_idx = 4
    splits = prepare_supervised_data(dataset)

    model = Sequential(
        [
            Linear(splits.x_train.ncol, 128),
            ReLU(),
            Linear(128, splits.y_train.ncol),
            Softmax(),
        ]
    )
    optimizer = SGD(model.parameters(), 0.1)

    epochs = 30
    progress = tqdm(range(epochs), desc="training")
    for _ in progress:
        optimizer.zero_grad()
        y_pred, weights = model.forward(splits.x_train, plot=True)
        loss = CCE(splits.y_train, y_pred)
        acc = accuracy(splits.y_train, y_pred)
        progress.set_postfix(loss=f"{loss.data[0]:.4f}", acc=f"{acc:.3f}")
        loss.backward()
        optimizer.step()

    # if weights:
    #     histogram(weights, "iris_weights.png")


if __name__ == "__main__":
    train_iris()
