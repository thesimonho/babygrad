"""Generate a number-words to digits sequence-to-sequence dataset.

Each example pairs an English spelling with its digit form:

    words                                digits
    three hundred forty two              342
    twelve thousand nine                 12009

The task is the Phase 8 transformer demonstration. It is deliberately small but
not memorisable: place value is a compositional rule, so a model that has only
memorised the rows it trained on will fail on unseen numbers, while one that has
learned the rule will not. Every number appears at most once in the file, so any
train/val/test split of it is automatically disjoint -- which is what makes that
distinction measurable.

The spelled form also gives the tokenizer real subword structure to find --
``-teen`` and ``-ty`` recur across the vocabulary, and ``hundred``/``thousand``
appear constantly -- which a purely symbolic task would not.

Splitting is the data pipeline's job, so this writes one undivided file.

Usage:

    python data/generate_number_words.py                # 1200 pairs
    python data/generate_number_words.py --pairs 100    # small dev run
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).with_name("number_words.csv")

DEFAULT_PAIRS = 1200
DEFAULT_MAX_DIGITS = 6
DEFAULT_SEED = 7

UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def spell(number: int) -> str:
    """Spell a non-negative integer below one million in English.

    Uses the American convention with no "and" separator, so every example has
    exactly one correct spelling and the target mapping stays unambiguous.
    """
    if number < 20:
        return UNITS[number]

    if number < 100:
        tens_word = TENS[number // 10]
        remainder = number % 10
        if remainder == 0:
            return tens_word
        return f"{tens_word} {UNITS[remainder]}"

    if number < 1000:
        hundreds_word = f"{UNITS[number // 100]} hundred"
        remainder = number % 100
        if remainder == 0:
            return hundreds_word
        return f"{hundreds_word} {spell(remainder)}"

    thousands_word = f"{spell(number // 1000)} thousand"
    remainder = number % 1000
    if remainder == 0:
        return thousands_word
    return f"{thousands_word} {spell(remainder)}"


def digit_length_ranges(max_digits: int) -> list[tuple[int, int]]:
    """Return the half-open integer range holding each digit length, shortest first."""
    ranges = [(0, 10)]
    for digits in range(2, max_digits + 1):
        ranges.append((10 ** (digits - 1), 10**digits))
    return ranges


def sample_numbers(
    pair_count: int,
    max_digits: int,
    random_number_generator: random.Random,
) -> list[int]:
    """Draw distinct numbers spread evenly across digit lengths.

    Sampling uniformly from the whole range would make ~88% of a six-digit
    dataset six digits long and omit one- and two-digit numbers entirely, so the
    model would only ever see one shape of problem. Balancing by length instead
    exposes the full compositional range and lets accuracy be reported per digit
    length, which is where a partly-learned place-value rule shows itself.

    Short lengths cannot fill an equal share -- there are only ten one-digit
    numbers -- so buckets are filled shortest first and each unused share flows
    on to the longer lengths.
    """
    ranges = digit_length_ranges(max_digits)
    total_available = sum(stop - start for start, stop in ranges)
    if pair_count > total_available:
        raise ValueError(
            f"requested {pair_count} pairs but only {total_available} "
            f"numbers below {max_digits} digits exist"
        )

    numbers: list[int] = []
    remaining_pairs = pair_count
    for index, (start, stop) in enumerate(ranges):
        remaining_buckets = len(ranges) - index
        fair_share = remaining_pairs // remaining_buckets
        take = min(stop - start, fair_share)
        numbers.extend(random_number_generator.sample(range(start, stop), take))
        remaining_pairs -= take

    random_number_generator.shuffle(numbers)
    return numbers


def build_rows(numbers: list[int]) -> list[dict[str, str]]:
    """Build the (words, digits) rows for a list of numbers."""
    return [{"words": spell(number), "digits": str(number)} for number in numbers]


def write_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write example rows to one CSV file."""
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["words", "digits"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def describe(rows: list[dict[str, str]]) -> None:
    """Report the row count and longest source/target, for sizing padding."""
    longest_words = max(len(row["words"].split()) for row in rows)
    longest_digits = max(len(row["digits"]) for row in rows)
    print(
        f"{len(rows)} rows   "
        f"longest source {longest_words} words   "
        f"longest target {longest_digits} digits"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse generator options, so a dev run can shrink the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--max-digits", type=int, default=DEFAULT_MAX_DIGITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Write the number-words dataset to disk as one undivided file."""
    arguments = parse_arguments()
    random_number_generator = random.Random(arguments.seed)

    numbers = sample_numbers(
        arguments.pairs, arguments.max_digits, random_number_generator
    )
    rows = build_rows(numbers)

    write_rows(OUTPUT_PATH, rows)
    describe(rows)


if __name__ == "__main__":
    main()
