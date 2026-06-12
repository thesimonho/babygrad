type Number = int | float
type Shape = tuple[int, ...]

type Tag = str
type Step = int
type History = dict[Tag, dict[Step, HistoryValue]]
type HistoryValue = float | list[float]
