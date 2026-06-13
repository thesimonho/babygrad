"""Shared type vocabulary: structural aliases and domain value types."""

type Number = int | float
type Shape = tuple[int, ...]

type Tag = str
type Step = int
type History = dict[Tag, dict[Step, HistoryValue]]
type HistoryValue = float | list[float]

