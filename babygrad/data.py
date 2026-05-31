import csv
from pathlib import Path
from .tensor import Tensor


def maybe_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def load_csv(path: Path, has_header: bool = True) -> Tensor:
    """Load a CSV file into a Tensor.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        Tensor: A Tensor containing the CSV data.
    """
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        nrow = 0
        ncol = 0
        output = []

        if has_header:
            _ = next(reader)

        for r in reader:
            nrow += 1
            ncol = len(r)
            output.extend([maybe_float(x) for x in r])

    return Tensor(output, (nrow, ncol))
