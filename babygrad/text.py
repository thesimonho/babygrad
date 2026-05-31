TRUNCATION_LIMIT = 8


def vector(data: list[object]) -> str:
    """Format a flat list of numbers with aligned truncation."""
    indexes = _visible_indexes(len(data))
    width = _vector_width(data, indexes)
    items = _format_vector(data, indexes, width)
    return f"[{items}]"


def matrix(
    data: list[object],
    nrow: int,
    ncol: int,
    headers: list[str] | None = None,
) -> str:
    """Format row-major data as an aligned matrix with optional headers."""
    if headers is not None and len(headers) != ncol:
        raise ValueError("headers length must match ncol")

    row_indexes = _visible_indexes(nrow)
    col_indexes = _visible_indexes(ncol)
    col_widths = _column_widths(data, ncol, row_indexes, col_indexes, headers)
    rows = [
        _format_matrix_row(data, ncol, row_index, col_indexes, col_widths)
        for row_index in row_indexes
    ]

    if headers is not None:
        rows.insert(0, _format_header(headers, col_indexes, col_widths))

    body = "\n".join(f"  {row}" for row in rows)
    return f"[\n{body}\n]"


def _format_header(
    headers: list[str],
    col_indexes: list[int | None],
    col_widths: dict[int, int],
) -> str:
    """Format one aligned header row for the matrix representation."""
    formatted_items = []

    for col_index in col_indexes:
        if col_index is None:
            formatted_items.append("...")
        else:
            formatted_items.append(headers[col_index].rjust(col_widths[col_index]))

    return "[" + "  ".join(formatted_items) + "]"


def _format_matrix_row(
    data: list[object],
    ncol: int,
    row_index: int | None,
    col_indexes: list[int | None],
    col_widths: dict[int, int],
) -> str:
    """Format one aligned row for the matrix representation."""
    if row_index is None:
        return "..."

    formatted_items = []

    for col_index in col_indexes:
        if col_index is None:
            formatted_items.append("...")
        else:
            value = data[row_index * ncol + col_index]
            width = col_widths[col_index]
            formatted_items.append(repr(value).rjust(width))

    return "[" + "  ".join(formatted_items) + "]"


def _format_vector(data: list[object], indexes: list[int | None], width: int) -> str:
    """Format scalar values with middle truncation and fixed width."""
    formatted_items = []

    for index in indexes:
        if index is None:
            formatted_items.append("...")
        else:
            formatted_items.append(repr(data[index]).rjust(width))

    return "  ".join(formatted_items)


def _column_widths(
    data: list[object],
    ncol: int,
    row_indexes: list[int | None],
    col_indexes: list[int | None],
    headers: list[str] | None = None,
) -> dict[int, int]:
    """Return display widths for the visible matrix columns."""
    widths = {}

    for col_index in col_indexes:
        if col_index is None:
            continue

        value_width = max(
            (
                len(repr(data[row_index * ncol + col_index]))
                for row_index in row_indexes
                if row_index is not None
            ),
            default=0,
        )
        header_width = len(headers[col_index]) if headers is not None else 0
        widths[col_index] = max(value_width, header_width)

    return widths


def _vector_width(data: list[object], indexes: list[int | None]) -> int:
    """Return the display width for visible vector values."""
    visible_indexes = [index for index in indexes if index is not None]

    if len(visible_indexes) == 0:
        return 1

    return max(len(repr(data[index])) for index in visible_indexes)


def _visible_indexes(length: int, limit: int = TRUNCATION_LIMIT) -> list[int | None]:
    """Return indexes to show, using None as the truncation marker."""
    if length <= limit:
        return list(range(length))

    head_count = limit // 2
    tail_count = limit - head_count
    head_indexes = list(range(head_count))
    tail_indexes = list(range(length - tail_count, length))
    return head_indexes + [None] + tail_indexes
