"""
AI written helpers for formatting data for display.
"""

from collections.abc import Sequence
from types import EllipsisType
from typing import TypeVar


TRUNCATION_LIMIT = 8
MAX_DECIMAL_PLACES = 4
ELLIPSIS = Ellipsis
ELLIPSIS_TEXT = "..."
T = TypeVar("T")


def vector(data: Sequence[object]) -> str:
    """Format a flat list of numbers with aligned truncation."""
    visible_items = _preview(data)
    visible_text = [_format_value(item) for item in visible_items]
    width = max((len(item) for item in visible_text), default=1)
    items = _align_items(visible_text, [width] * len(visible_text))
    return f"[{items}]"


def matrix(
    data: Sequence[object],
    nrow: int,
    ncol: int,
    headers: Sequence[str] | None = None,
) -> str:
    """Format row-major data as an aligned matrix with optional headers."""
    if headers is not None and len(headers) != ncol:
        raise ValueError("headers length must match ncol")

    row_indexes = _preview(list(range(nrow)))
    col_indexes = _preview(list(range(ncol)))
    col_widths = _column_widths(data, ncol, row_indexes, col_indexes, headers)
    rows = []

    if headers is not None:
        header_items = _row_items(headers, 1, 0, col_indexes)
        header_text = [_format_value(item, is_header=True) for item in header_items]
        rows.append("[" + _align_items(header_text, col_widths) + "]")

    for row_index in row_indexes:
        if isinstance(row_index, EllipsisType):
            rows.append(ELLIPSIS_TEXT)
            continue

        row_items = _row_items(data, ncol, row_index, col_indexes)
        row_text = [_format_value(item) for item in row_items]
        rows.append("[" + _align_items(row_text, col_widths) + "]")

    body = "\n".join(f"  {row}" for row in rows)
    return f"[\n{body}\n]"


def _align_items(items: list[str], widths: list[int]) -> str:
    """Right-align visible text items."""
    return "  ".join(item.rjust(widths[index]) for index, item in enumerate(items))


def _row_items(
    data: Sequence[object],
    ncol: int,
    row_index: int,
    col_indexes: Sequence[int | EllipsisType],
) -> list[object | EllipsisType]:
    """Return visible row values with an ellipsis marker if needed."""
    items = []

    for col_index in col_indexes:
        if isinstance(col_index, EllipsisType):
            items.append(ELLIPSIS)
            continue

        items.append(data[row_index * ncol + col_index])

    return items


def _column_widths(
    data: Sequence[object],
    ncol: int,
    row_indexes: Sequence[int | EllipsisType],
    col_indexes: Sequence[int | EllipsisType],
    headers: Sequence[str] | None = None,
) -> list[int]:
    """Return display widths for the visible matrix columns."""
    widths = []

    for col_index in col_indexes:
        if isinstance(col_index, EllipsisType):
            widths.append(len(ELLIPSIS_TEXT))
            continue

        visible_widths = []
        if headers is not None:
            visible_widths.append(
                len(_format_value(headers[col_index], is_header=True))
            )

        for row_index in row_indexes:
            if not isinstance(row_index, EllipsisType):
                value = data[row_index * ncol + col_index]
                visible_widths.append(len(_format_value(value)))

        widths.append(max(visible_widths, default=1))

    return widths


def _format_value(value: object, is_header: bool = False) -> str:
    """Return the display text for one visible value."""
    if isinstance(value, EllipsisType):
        return ELLIPSIS_TEXT
    if is_header:
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    return repr(value)


def _format_float(value: float) -> str:
    """Limit float output so previews stay narrow."""
    text = f"{value:.{MAX_DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    if "." not in text:
        text += ".0"
    return text


def _preview(
    items: Sequence[T],
    limit: int = TRUNCATION_LIMIT,
) -> list[T | EllipsisType]:
    """Return the visible edges of a list, with an ellipsis in the middle."""
    if len(items) <= limit:
        return list(items)

    head_count = limit // 2
    tail_count = limit - head_count
    return list(items[:head_count]) + [ELLIPSIS] + list(items[-tail_count:])
