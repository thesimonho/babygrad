import csv
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from babygrad import formatting
from babygrad.tensor import Tensor


class Sample(NamedTuple):
    """A single row pair of targets and features."""

    features: list
    target: list


class Batch(NamedTuple):
    """A pair of target and feature tensors."""

    features: Tensor
    target: Tensor


class Dataset(ABC):
    """
    Loads, or stores a reference to, data on disk.
    """

    data: list[Sample]

    def __init__(self, path: Path):
        self.path = path

    @property
    def nrow(self) -> int:
        return len(self.data)

    @property
    def ncol(self) -> int:
        if not self.data:
            return 0
        x, y = self.data[0]
        return len(x) + len(y)

    def __len__(self):
        return self.nrow

    @abstractmethod
    def __getitem__(self, i) -> Sample:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    def _flat_rows(self) -> list:
        """Flatten each (x, y) row into row-major cells for matrix display."""
        cells = []
        for x, y in self.data:
            cells.extend(x)
            cells.extend(y)
        return cells

    def _maybe_float(self, value):
        try:
            return float(value)
        except ValueError:
            return value


class CSVDataset(Dataset):
    headers: Sample | None = None
    target_col_idx: int

    def __init__(self, path: Path, target_col_idx: int, has_header: bool = True):
        super().__init__(path)
        self.has_header = has_header
        self.target_col_idx = target_col_idx
        self.load(has_header)

    def __repr__(self) -> str:
        """Return an aligned matrix-style preview of the dataset rows."""
        headers = None
        if self.headers is not None:
            x_headers, y_headers = self.headers
            headers = [*x_headers, *y_headers]
        return f"{self.nrow} rows x {self.ncol} cols\n{formatting.matrix(self._flat_rows(), self.nrow, self.ncol, headers)}"

    def __getitem__(self, i) -> Sample:
        assert isinstance(i, int), "Dataset only supports integer indexing"
        assert self.target_col_idx is not None
        return self.data[i]

    def load(self, has_header: bool = True) -> None:
        with open(self.path) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            if has_header:
                x, y = self._split_feature_target(next(reader))
                self.headers = Sample(features=x, target=y)
            else:
                self.headers = None

            output = []
            for r in reader:
                x, y = self._split_feature_target(r)
                for i in range(len(x)):
                    x[i] = self._maybe_float(x[i])
                for i in range(len(y)):
                    y[i] = self._maybe_float(y[i])
                output.append(Sample(x, y))
            self.data = output

    def _split_feature_target(self, row) -> Sample:
        """Split a row into (x, y) tuple"""
        if self.target_col_idx is None:
            raise ValueError("Target column has not been set")

        x = list(row)
        y = [x.pop(self.target_col_idx)]
        return Sample(x, y)


@dataclass
class DataSplit:
    rows: list[Sample]
    one_hot_mapping: dict | None = None

    def __getitem__(self, i):
        return self.rows[i]

    def __len__(self):
        return len(self.rows)

    @property
    def nrow(self) -> int:
        return len(self.rows)

    @property
    def n_features(self) -> int:
        return len(self.rows[0].features)

    @property
    def n_targets(self) -> int:
        if self.one_hot_mapping is None:
            return len(self.rows[0].target)

        return len(self.one_hot_mapping[self.rows[0].target[0]])


class DataLoader:
    """Turns data into tensors and creates minibatches"""

    def __init__(self, data: DataSplit, batch_size: int | None = None) -> None:
        self.data = data
        self.batch_size = batch_size if batch_size is not None else len(data)

    def __iter__(self):
        shuffled = self.data.rows.copy()
        random.shuffle(shuffled)

        for row in range(0, len(shuffled), self.batch_size):
            yield self.convert_to_tensor(shuffled[row : row + self.batch_size])

    def convert_to_tensor(self, rows: list[Sample]) -> Batch:
        """Collate a list of Samples into one Batch of tensors.

        Currently assumes flat, fixed-width tabular features and a classification/regression target.
        Need to inject a collation fn for different types of data. Probably needs to accept config too.
        """
        x = []
        y = []
        y_cols = 1

        for row in rows:
            x.extend(row.features)

            if self.data.one_hot_mapping is not None:
                one_hot_vector = self.data.one_hot_mapping[row.target[0]]
                y.extend(self.data.one_hot_mapping[row.target[0]])
                y_cols = len(one_hot_vector)
            else:
                y.extend(row.target)
                y_cols = len(row.target)

        features = Tensor(x, shape=(len(rows), len(rows[0].features)))
        target = Tensor(y, shape=(len(rows), y_cols))
        return Batch(features, target)

    def full_batch(self) -> Batch:
        return next(iter(self))


def split_train_val_test(
    dataset: Dataset, train_prop=0.8, val_prop=0.1, test_prop=0.1, one_hot: bool = False
) -> tuple[DataSplit, DataSplit, DataSplit]:
    """Splits data into train, val, test raw data sets"""
    assert math.isclose(train_prop + val_prop + test_prop, 1.0)

    shuffled = random.sample(dataset.data, dataset.nrow)
    train_end = int(dataset.nrow * train_prop)
    val_end = train_end + int(dataset.nrow * val_prop)

    data_train = DataSplit(rows=shuffled[:train_end])
    data_val = DataSplit(rows=shuffled[train_end:val_end])
    data_test = DataSplit(rows=shuffled[val_end:])

    if one_hot:
        one_hot_mapping = encode_one_hot(data_train)
        data_train.one_hot_mapping = one_hot_mapping
        data_val.one_hot_mapping = one_hot_mapping
        data_test.one_hot_mapping = one_hot_mapping

    return (data_train, data_val, data_test)


def encode_one_hot(split: DataSplit) -> dict[str, list[int]]:
    """Encode a split's targets into one-hot vectors.
    Split should be the training data to avoid leaking data."""
    mapping = {}
    for row in split.rows:
        target = row.target[0]
        if target in mapping:
            continue
        mapping[target] = []

    n_classes = len(mapping)
    for i, key in enumerate(mapping.keys()):
        mapping[key] = [0] * n_classes
        mapping[key][i] = 1

    return mapping
