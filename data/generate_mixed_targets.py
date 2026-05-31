"""Generate the mixed-target CSV dataset.

The generated rows have four numeric inputs, one categorical target, and one
continuous target. The formulas are intentionally simple so the dataset is easy
to inspect while building forward-only network code.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).with_name("mixed_targets.csv")
ROW_COUNT = 60
SEED = 7


def classify_raw_score(raw_score: float) -> str:
    """Return a categorical target from the synthetic raw score."""
    if raw_score < 0.15:
        return "low"

    if raw_score < 0.45:
        return "medium"

    return "high"


def build_row(index: int, random_number_generator: random.Random) -> dict[str, object]:
    """Build one deterministic synthetic example."""
    x1 = round((index % 10) / 9, 3)
    x2 = round(((index * 3) % 17) / 16, 3)
    x3 = round(((index * 7 + 5) % 23) / 22, 3)
    x4 = round(random_number_generator.uniform(-1.0, 1.0), 3)

    raw_score = 0.8 * x1 - 0.5 * x2 + 0.35 * x3 + 0.2 * x4
    continuous_target = round(10 + 4 * x1 - 2 * x2 + 1.5 * x3 + 0.75 * x4, 3)

    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "category_target": classify_raw_score(raw_score),
        "continuous_target": continuous_target,
    }


def main() -> None:
    """Write the mixed-target dataset to disk."""
    random_number_generator = random.Random(SEED)
    fieldnames = ["x1", "x2", "x3", "x4", "category_target", "continuous_target"]

    with OUTPUT_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(ROW_COUNT):
            writer.writerow(build_row(index, random_number_generator))


if __name__ == "__main__":
    main()
