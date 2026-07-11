"""
Schedulers that control the learning rate over the course of a training run.
"""

import math
from typing import Protocol, Generator


class Scheduler(Protocol):
    def __call__(self, epoch: int) -> float: ...


class CosineAnnealingLR:
    """
    Reset the LR after a run of X epochs.
    Within a run, the LR decays as a function of cosine.
    """

    def __init__(
        self,
        T_0: int = 1,
        T_mult: int = 2,
        lr_range: tuple[float, float] = (0.0, 1.0),
    ):
        """Decay and reset the LR by cosine annealing.

        Example: T_0 = 1, T_mult = 2 gives runs with these epoch sizes:
        [1,2,4,8,16,...]

        After 1 epoch, the LR is reset for the next run. The next run contains 2 epochs etc.

        Args:
            T_0: Number of epochs in the first run.
            T_mult: The size of each subsequent run increases by this multipler.
            lr_range: The range of values that LR is allowed to take on.
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.lr_min, self.lr_max = lr_range

    def __call__(self, epoch: int) -> float:
        run_lengths = self._get_run_lengths(self.T_0, self.T_mult)
        lower = 0

        while True:
            current_size = next(run_lengths)
            # the restart schedule provides the length of each run
            # but epoch is cumulative across the entire training
            # need to convert run length items to their corresponding epoch id
            upper = lower + current_size
            if epoch > upper:
                lower = upper + 1
                continue
            break

        # T_cur is the number of epochs since the last restart
        T_cur = epoch - lower
        progress = T_cur / current_size

        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(progress * math.pi)
        )
        return lr

    def _get_run_lengths(self, T_0, T_mult) -> Generator[int]:
        """
        Return the number of epochs in each run.
        A restart occurs after the end of a run.
        """
        period = T_0
        while True:
            yield period
            period *= T_mult


class ConstantLR:
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, epoch: int) -> float:
        return self.lr
