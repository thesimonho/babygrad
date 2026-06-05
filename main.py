from pathlib import Path
from babygrad.data import load_csv, prepare_supervised_data
from babygrad.nn import CCE, ReLU, Sequential, Linear, Softmax
from babygrad.plot import histogram


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
    y_pred, weights = model.forward(splits.x_train, plot=True)
    print(y_pred)
    loss = CCE(splits.y_train, y_pred)
    print(loss)

    if weights:
        histogram(weights, "iris_weights.png")


if __name__ == "__main__":
    train_iris()
