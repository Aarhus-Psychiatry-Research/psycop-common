from collections.abc import Iterable
from functools import partial
from typing import Callable

import torch
from torch.optim.lr_scheduler import _LRScheduler  # , # type: ignore
from transformers import get_linear_schedule_with_warmup

from .registry import SequenceRegistry

OptimizerFn = Callable[[Iterable[torch.nn.parameter.Parameter]], torch.optim.Optimizer]
LRSchedulerFn = Callable[[torch.optim.Optimizer], _LRScheduler]


def _configure_adam(
    parameters,  # noqa: ANN001, # type: ignore
    lr,  # noqa: ANN001, # type: ignore
) -> torch.optim.Optimizer:
    return torch.optim.Adam(parameters, lr=lr)


def _configure_adamw(
    parameters,  # noqa: ANN001, # type: ignore
    lr,  # noqa: ANN001, # type: ignore
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=lr)


@SequenceRegistry.optimizers.register("adam")
def create_adam(lr: float) -> OptimizerFn:
    return partial(_configure_adam, lr=lr)


@SequenceRegistry.optimizers.register("adamw")
def create_adamw(lr: float) -> OptimizerFn:
    return partial(_configure_adamw, lr=lr)


def _configure_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> _LRScheduler:
    return get_linear_schedule_with_warmup(  # type: ignore
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch,
    )


@SequenceRegistry.lr_schedulers.register("linear_schedule_with_warmup")
def create_linear_schedule_with_warmup(
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LRSchedulerFn:
    return partial(
        _configure_linear_schedule_with_warmup,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch,
    )
