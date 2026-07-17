from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, DefaultDict

from tqdm import tqdm

from babygrad.data import DataLoader
from babygrad.recorder import Recorder
from babygrad.state import _is_training, bound
from babygrad.tensor import Tensor
from babygrad.types import Number, Scope

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
        # explicit containment tree, built as we stamp: each module becomes a Scope
        # linked to its outer scope, so consumers read structure instead of
        # re-splitting the name string.
        self.scopes: dict[str, Scope] = {}
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
        leaf = f"{root.name}_{idx}"
        root.name = f"{prefix}/{leaf}" if prefix else leaf
        self.scopes[root.name] = Scope(
            id=root.name,
            label=leaf,
            outer_scope=prefix or None,
            collapsed=root.collapse,
        )
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
        assert n_batches > 0, ValueError("Training data contains no rows/batches.")

        reporter = TrainingReporter(self.recorder, self.config, n_batches)

        for e in range(self.config.epochs):
            reporter.start_epoch(e)
            self.config.optimizer.lr = self.config.scheduler(e)

            # training
            batches = DataLoader(train, self.config.batch_size)
            for x_train, y_train in batches:
                pred_train, train_loss_tensor = self._train_one_batch(x_train, y_train)
                reporter.on_batch(y_train, pred_train, train_loss_tensor)

            # validation
            validation_loss_value, _ = self.evaluate(val)
            reporter.end_epoch(validation_loss_value)

        train_loss_tensor = reporter.loss_tensor
        reporter.close()
        return train_loss_tensor

    def _train_one_batch(
        self, x_train: Tensor, y_train: Tensor
    ) -> tuple[Tensor, Tensor]:
        self.config.optimizer.zero_grad()
        pred_train = self.model.forward(x_train)
        with bound(_is_training, True):
            loss_tensor = self.config.criterion.forward(y_train, pred_train)

        loss_tensor.backward()
        self.config.optimizer.step()

        return pred_train, loss_tensor

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


class TrainingReporter:
    def __init__(self, recorder: Recorder | None, config: TrainConfig, n_batches: int):
        self.recorder = recorder
        self.config = config
        self.batch_sizes: list[int] = []
        self.accum_loss: list[Number] = []
        self.epoch_metrics: DefaultDict[str, list[Number]] = defaultdict(list)
        self.loss_tensor: Tensor | None = None

        self.progress_epoch = tqdm(
            total=self.config.epochs, desc="train (epochs)", leave=True, position=0
        )
        self.progress_batch = tqdm(
            total=n_batches,
            desc="train (batch)",
            leave=False,
            position=1,
        )

    def start_epoch(self, epoch: int) -> None:
        self.progress_batch.reset()
        self.loss_tensor = None
        self.batch_sizes = []
        self.accum_loss = []
        self.epoch_metrics: DefaultDict[str, list[Number]] = defaultdict(list)

        if self.recorder is None:
            return
        self.recorder.step = epoch

    def on_batch(self, y: Tensor, pred: Tensor, loss: Tensor) -> None:
        self.loss_tensor = loss
        self.batch_sizes.append(y.nrow)
        self.accum_loss.append(loss.data[0])

        if self.config.metrics is not None:
            for metric in self.config.metrics:
                self.epoch_metrics[metric.name].append(metric.compute(y, pred))

        self.progress_batch.set_postfix(
            {
                name: f"{self._batch_weighted_mean(values):.3f}"
                for name, values in self.epoch_metrics.items()
            },
            loss=f"{loss.data[0]:.3f}",
        )
        self.progress_batch.update()

    def end_epoch(self, validation_loss_value: Number) -> None:
        if self.recorder is None:
            return

        if self.loss_tensor is not None:
            self.recorder.capture(root=self.loss_tensor)
        self.recorder.record("loss", self._batch_weighted_mean(self.accum_loss))
        self.recorder.record("val_loss", validation_loss_value)
        if self.config.metrics is not None:
            for name, values in self.epoch_metrics.items():
                self.recorder.record(name, self._batch_weighted_mean(values))

        self.progress_epoch.update()

    def close(self) -> None:
        self.progress_batch.close()
        self.progress_epoch.close()

    def _batch_weighted_mean(self, values: list[Number]) -> float:
        """
        Calculate a mean weighted by the size of each batch.
        """
        assert len(values) > 0 and len(self.batch_sizes) > 0
        assert len(values) == len(self.batch_sizes)
        return sum(v * n for v, n in zip(values, self.batch_sizes)) / sum(
            self.batch_sizes
        )
