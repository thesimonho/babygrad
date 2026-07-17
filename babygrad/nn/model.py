from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, DefaultDict

from tqdm import tqdm

from babygrad.data import DataLoader
from babygrad.recorder import Recorder
from babygrad.state import _is_training, bound
from babygrad.tensor import Tensor
from babygrad.types import Number

if TYPE_CHECKING:
    from babygrad.data import DataSplit
    from babygrad.metrics import Metric
    from babygrad.nn.losses import Loss
    from babygrad.nn.modules import Module
    from babygrad.nn.optimizers import Optimizer
    from babygrad.nn.schedulers import Scheduler


@dataclass(slots=True)
class TrainConfig:
    epochs: int
    batch_size: int
    optimizer: Optimizer
    scheduler: Scheduler
    criterion: Loss
    metrics: list[Metric] | None = None


class Model:
    def __init__(self, root: Module):
        self.root = root
        # scopes of modules asking to be drawn as one box. collected here because this
        # walk is the only place a live Module and its final scope string coexist.
        self.collapsed_scopes: set[str] = set()
        self.stamp_name_and_scope(self.root)

    def eval(self, x: Tensor) -> Tensor:
        """
        Use for forward pass inference, not training.
        """
        with bound(_is_training, False):
            return self.root.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        with bound(_is_training, True):
            return self.root.forward(x)

    def stamp_name_and_scope(self, root: Module, prefix: str = "", idx: int = 0):
        """
        Set the name and scope of all descendent Modules and their children
        """
        root.name = f"{prefix + '/' if prefix else ''}{root.name}_{idx}"
        if root.collapse:
            self.collapsed_scopes.add(root.name)

        for idx, c in enumerate(root.children()):
            self.stamp_name_and_scope(c, root.name, idx)

        for idx, p in enumerate(root.own_parameters()):
            p.scope = root.name
            p.name = f"{root.name.split('/')[-1]}/{p.name}"


class Trainer:
    def __init__(
        self, model: Model, config: TrainConfig, recorder: Recorder | None = None
    ):
        self.model = model
        self.config = config
        self.recorder = recorder

    def fit(self, train: DataSplit, val: DataSplit) -> Tensor | None:
        """
        Train the model on the train split, validating on the val split.
        """

        n_batches = ceil(train.nrow / self.config.batch_size)
        progress_epoch = tqdm(range(self.config.epochs), desc="train (epochs)")
        progress_batch = tqdm(
            total=n_batches,
            desc="train (batch)",
            leave=False,
            position=1,
        )

        train_loss: Tensor | None = None

        for e in progress_epoch:
            progress_batch.reset()
            self.config.optimizer.lr = self.config.scheduler(e)
            if self.recorder is not None:
                self.recorder.step = e

            accum_loss = []
            epoch_metrics: DefaultDict[str, list[Number]] = defaultdict(list)
            batches = DataLoader(train, self.config.batch_size)
            batch_sizes = []

            for x_train, y_train in batches:
                self.config.optimizer.zero_grad()

                pred_train = self.model.forward(x_train)
                with bound(_is_training, True):
                    train_loss = self.config.criterion.forward(y_train, pred_train)

                batch_sizes.append(y_train.nrow)
                accum_loss.append(train_loss.data[0])

                if self.config.metrics is not None:
                    for metric in self.config.metrics:
                        epoch_metrics[metric.name].append(
                            metric.compute(y_train, pred_train)
                        )

                train_loss.backward()
                self.config.optimizer.step()

                progress_batch.set_postfix(
                    {
                        name: f"{weighted_mean(values, batch_sizes):.3g}"
                        for name, values in epoch_metrics.items()
                    },
                    loss=f"{train_loss.data[0]:.3g}",
                )
                progress_batch.update()

            # validation
            validation_loss, _ = self.evaluate(val)

            if self.recorder is not None:
                if train_loss is not None:
                    self.recorder.capture(root=train_loss)
                self.recorder.record("loss", weighted_mean(accum_loss, batch_sizes))
                self.recorder.record("val_loss", validation_loss)
                if self.config.metrics is not None:
                    for name, values in epoch_metrics.items():
                        self.recorder.record(name, weighted_mean(values, batch_sizes))

        progress_batch.close()
        return train_loss

    def test(self, test: DataSplit) -> None:
        """
        Evaluate the model on the test split, print metrics and loss.
        """
        loss, metrics = self.evaluate(test)
        print(f"\ntest loss: {loss:.3g}")
        print(
            ", ".join([f"test {name}: {value:.3g}" for name, value in metrics.items()])
        )

    def evaluate(self, split: DataSplit) -> tuple[Number, dict[str, list[Number]]]:
        """
        Evaluate the model on a split, return loss and metrics.
        """
        x, y = DataLoader(split).full_batch()
        pred = self.model.eval(x)
        with bound(_is_training, False):
            loss = self.config.criterion.forward(y, pred)

        metrics_output = {}
        if self.config.metrics is not None:
            for metric in self.config.metrics:
                metrics_output[metric.name] = metric.compute(y, pred)

        return loss.data[0], metrics_output


def weighted_mean(values: list[Number], batch_sizes: list[int]) -> float:
    assert len(values) > 0 and len(batch_sizes) > 0
    assert len(values) == len(batch_sizes)
    return sum(v * n for v, n in zip(values, batch_sizes)) / sum(batch_sizes)
