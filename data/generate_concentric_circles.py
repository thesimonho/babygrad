"""Generate a concentric-circles classification dataset.

The two classes form an inner disk and an outer ring separated by an empty
margin. No linear boundary can separate them, and recovering the true boundary
requires composing a radius-like feature (``x1**2 + x2**2``) from the raw
inputs. That makes the task a clean demonstration target for Phase 7: a shallow
network underfits, a deep plain stack struggles to train, and a deep residual
network learns the ring boundary cleanly.

Each row has two numeric inputs and one string class label:

    x1, x2, label   (label in {"inner", "outer"})
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).with_name("concentric_circles.csv")
ROW_COUNT = 5000
SEED = 7

# Inner disk fills radius [0, INNER_RADIUS]; the outer ring fills
# [OUTER_INNER_RADIUS, OUTER_OUTER_RADIUS]. The gap between INNER_RADIUS and
# OUTER_INNER_RADIUS is the empty margin that keeps the signal clear.
INNER_RADIUS = 1.0
OUTER_INNER_RADIUS = 1.5
OUTER_OUTER_RADIUS = 2.5

# Small jitter added to each coordinate so the rings are not perfectly clean
# while remaining clearly separated.
NOISE_STANDARD_DEVIATION = 0.05


def sample_point(
    minimum_radius: float,
    maximum_radius: float,
    random_number_generator: random.Random,
) -> tuple[float, float]:
    """Sample one (x1, x2) point in a ring between two radii.

    The radius uses ``sqrt`` of a uniform draw so points spread evenly across
    the ring's area instead of clustering near the inner edge.
    """
    lower_area = minimum_radius**2
    upper_area = maximum_radius**2
    radius = math.sqrt(random_number_generator.uniform(lower_area, upper_area))
    angle = random_number_generator.uniform(0.0, 2.0 * math.pi)

    x1 = radius * math.cos(angle) + random_number_generator.gauss(0.0, NOISE_STANDARD_DEVIATION)
    x2 = radius * math.sin(angle) + random_number_generator.gauss(0.0, NOISE_STANDARD_DEVIATION)
    return round(x1, 4), round(x2, 4)


def build_rows(random_number_generator: random.Random) -> list[dict[str, object]]:
    """Build a balanced, shuffled list of inner and outer example rows."""
    examples_per_class = ROW_COUNT // 2
    rows: list[dict[str, object]] = []

    for _ in range(examples_per_class):
        x1, x2 = sample_point(0.0, INNER_RADIUS, random_number_generator)
        rows.append({"x1": x1, "x2": x2, "label": "inner"})

    for _ in range(examples_per_class):
        x1, x2 = sample_point(OUTER_INNER_RADIUS, OUTER_OUTER_RADIUS, random_number_generator)
        rows.append({"x1": x1, "x2": x2, "label": "outer"})

    random_number_generator.shuffle(rows)
    return rows


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    """Write all example rows to one CSV file."""
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["x1", "x2", "label"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Write the concentric-circles dataset to disk."""
    random_number_generator = random.Random(SEED)
    rows = build_rows(random_number_generator)
    write_rows(OUTPUT_PATH, rows)


if __name__ == "__main__":
    main()
