import csv
import random
from dataclasses import dataclass, replace
from typing import Iterator
from pathlib import Path

from .tensor import Tensor
from . import formatting


@dataclass
class Dataset:
    rows: list[list]
    headers: list[str] | None = None
    target_col_idx: int | None = None
    one_hot: bool = False
    one_hot_mapping: dict | None = None

    @property
    def nrow(self) -> int:
        return len(self.rows)

    @property
    def ncol(self) -> int:
        if len(self.rows) == 0:
            return len(self.headers or [])

        return len(self.rows[0])

    def __repr__(self) -> str:
        """Return an aligned matrix-style preview of the dataset rows."""
        return f"{self.nrow} rows x {self.ncol} cols\n{formatting.matrix(self.flat_rows(), self.nrow, self.ncol, self.headers)}"

    def flat_rows(self) -> list:
        """Return the nested rows as one row-major list."""
        output = []

        for row in self.rows:
            output.extend(row)

        return output


@dataclass
class DataSplit:
    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Tensor
    y_test: Tensor


def _maybe_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def load_csv(path: Path, has_header: bool = True) -> Dataset:
    """Load a CSV file into lists.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        List: data as nested lists
    """
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        headers = next(reader) if has_header else None
        output = []

        for r in reader:
            output.append([_maybe_float(x) for x in r])

    return Dataset(headers=headers, rows=output)


def split_train_val_test(data: Dataset, train_prop=0.8, val_prop=0.1, test_prop=0.1):
    assert train_prop + val_prop + test_prop == 1.0

    shuffled = random.sample(data.rows, data.nrow)
    train_end = int(data.nrow * train_prop)
    val_end = train_end + int(data.nrow * val_prop)

    data_train = replace(data, rows=shuffled[:train_end])
    data_val = replace(data, rows=shuffled[train_end:val_end])
    data_test = replace(data, rows=shuffled[val_end:])

    if data.target_col_idx is not None and data.one_hot:
        mapping = {}
        for row in data_train.rows:
            target = row[data.target_col_idx]
            if target in mapping:
                continue
            mapping[target] = []

        n_classes = len(mapping)
        for i, key in enumerate(mapping.keys()):
            mapping[key] = [0] * n_classes
            mapping[key][i] = 1

        data_train.one_hot_mapping = mapping
        data_val.one_hot_mapping = mapping
        data_test.one_hot_mapping = mapping

    return data_train, data_val, data_test


def split_feature_target(data: Dataset):
    if data.target_col_idx is None:
        raise ValueError("Target column has not been set")

    if data.headers:
        x_header = list(data.headers)
        y_header = [x_header.pop(data.target_col_idx)]
    else:
        x_header = None
        y_header = None

    x = []
    y = []
    for row in data.rows:
        row_x = list(row)
        y_value = row_x.pop(data.target_col_idx)

        if data.one_hot_mapping is not None and data.one_hot:  # one hot encode
            try:
                row_y = data.one_hot_mapping[y_value]
            except KeyError:
                raise ValueError(
                    "Dataset contains an unseen target label. Check your dataset shuffling."
                )
        else:  # regression
            row_y = [y_value]

        y.append(row_y)
        x.append(row_x)

    return Dataset(rows=x, headers=x_header, target_col_idx=None), Dataset(
        rows=y, headers=y_header
    )


def to_tensor(data: Dataset) -> Tensor:
    return Tensor(data.flat_rows(), shape=(data.nrow, data.ncol))


def prepare_supervised_data(data: Dataset):
    """
    Helper function to combine loading and splitting data into tensors.
    """
    train, val, test = split_train_val_test(data)
    x_train, y_train = split_feature_target(train)
    x_val, y_val = split_feature_target(val)
    x_test, y_test = split_feature_target(test)
    t_train = to_tensor(x_train)
    t_val = to_tensor(x_val)
    t_test = to_tensor(x_test)
    t_train_target = to_tensor(y_train)
    t_val_target = to_tensor(y_val)
    t_test_target = to_tensor(y_test)

    return DataSplit(
        x_train=t_train,
        y_train=t_train_target,
        x_val=t_val,
        y_val=t_val_target,
        x_test=t_test,
        y_test=t_test_target,
    )


def create_minibatches(
    x: Tensor, y: Tensor, batch_size: int = 32
) -> Iterator[tuple[Tensor, Tensor]]:
    idx_ordered = list(range(x.nrow))
    random.shuffle(idx_ordered)
    random_x = _get_data_by_idx(x, idx_ordered)
    random_y = _get_data_by_idx(y, idx_ordered)

    assert len(random_x) == len(random_y)

    for i in range(0, random_x.nrow, batch_size):
        yield random_x[i : i + batch_size], random_y[i : i + batch_size]


def _get_data_by_idx(data: Tensor, order: list[int]) -> Tensor:
    reordered = [data[i : i + 1] for i in order]
    flat = []
    for row in reordered:
        flat.extend(row.data)
    return Tensor(flat, (len(reordered), data.ncol))
