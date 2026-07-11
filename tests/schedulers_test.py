import math

import pytest

from babygrad.nn.schedulers import ConstantLR, CosineAnnealingLR


def _curve(scheduler, n_epochs):
    """LR at each epoch 0..n_epochs-1, asked in order (the training-loop usage)."""
    return [scheduler(epoch) for epoch in range(n_epochs)]


def _epochs_at(lrs, target):
    """Epochs whose LR equals `target` — used to locate run starts / troughs."""
    return [e for e, lr in enumerate(lrs) if math.isclose(lr, target, abs_tol=1e-9)]


def test_every_run_starts_at_lr_max():
    # cos(0) = 1, so the first epoch of a run sits at the top of the range.
    scheduler = CosineAnnealingLR(T_0=3, T_mult=2, lr_range=(0.1, 0.9))
    assert scheduler(0) == pytest.approx(0.9)


def test_lr_never_leaves_the_range():
    # 0.5 * (1 + cos θ) ∈ [0, 1], so lr stays inside [lr_min, lr_max] for any epoch.
    lr_min, lr_max = 0.1, 0.9
    scheduler = CosineAnnealingLR(T_0=3, T_mult=2, lr_range=(lr_min, lr_max))
    for lr in _curve(scheduler, 50):
        assert lr_min - 1e-9 <= lr <= lr_max + 1e-9


def test_curve_descend_to_near_lr_min():
    # By the end of a run the cosine has swung to (near) the bottom of the range.
    scheduler = CosineAnnealingLR(T_0=2, T_mult=2, lr_range=(0.0, 1.0))
    assert min(_curve(scheduler, 30)) < 0.05


def test_warm_restart_return_to_lr_max():
    # A restart jumps lr back to the top, so lr_max is hit more than just at epoch 0.
    scheduler = CosineAnnealingLR(T_0=2, T_mult=2, lr_range=(0.0, 1.0))
    run_starts = _epochs_at(_curve(scheduler, 30), 1.0)
    assert len(run_starts) >= 3


def test_restart_periods_grow():
    # With T_mult > 1 each run is longer than the last, so the gaps between
    # successive run-starts strictly increase.
    scheduler = CosineAnnealingLR(T_0=2, T_mult=2, lr_range=(0.0, 1.0))
    run_starts = _epochs_at(_curve(scheduler, 30), 1.0)
    gaps = [b - a for a, b in zip(run_starts, run_starts[1:])]
    assert len(gaps) >= 2
    assert all(later > earlier for earlier, later in zip(gaps, gaps[1:]))


def test_is_a_pure_function_of_epoch():
    scheduler = CosineAnnealingLR(T_0=2, T_mult=2, lr_range=(0.0, 1.0))

    # idempotent: asking for the same epoch twice returns the same lr
    assert scheduler(5) == scheduler(5)

    # order-independent: an epoch's lr does not depend on what was asked before it
    forward = {e: scheduler(e) for e in range(20)}
    backward = {e: scheduler(e) for e in reversed(range(20))}
    assert forward == backward


def test_constant_lr_is_flat():
    scheduler = ConstantLR(0.3)
    assert scheduler(0) == 0.3
    assert scheduler(1000) == 0.3
