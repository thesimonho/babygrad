from collections import defaultdict
from babygrad.aliases import History, HistoryValue, Step, Tag


class Recorder:
    """Collects tagged training data, keyed by the loop-supplied step."""

    def __init__(self):
        self.step: Step = 0
        self.history: History = defaultdict(dict)

    def set_step(self, step: Step):
        self.step = step

    def record(self, tag: Tag, value: HistoryValue):
        self.history[tag][self.step] = value
