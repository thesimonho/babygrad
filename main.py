from pathlib import Path
from babygrad.data import (
    load_csv,
    prepare_supervised_data,
)


def main():
    dataset = load_csv(Path("./data/iris.csv"))
    dataset.one_hot = True
    dataset.target_col_idx = 4
    splits = prepare_supervised_data(dataset)

    print(splits)


if __name__ == "__main__":
    main()
