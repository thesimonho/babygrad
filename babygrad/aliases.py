type Number = int | float
type Shape = tuple[int, ...]

type Tag = str
type RelativeTag = str
type Step = int
type Report = dict[RelativeTag, HistoryValue]
type History = dict[Tag, dict[Step, HistoryValue]]
type HistoryValue = float | list[float]
