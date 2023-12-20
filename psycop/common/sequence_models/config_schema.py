"""
The config Schema for sequence models.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger as plLogger
from pydantic import BaseModel

from psycop.common.feature_generation.sequences.prediction_time_collater import BasePredictionTimeCollater

from ..feature_generation.sequences.patient_slice_collater import (
    BasePatientSliceCollater,
)
from .tasks.patientslice_classifier_base import BasePatientSliceClassifier
from .tasks.pretrainer_base import BasePatientSlicePretrainer


class TrainerConfigSchema(BaseModel):
    """
    A config for the pytorch lightning trainer
    """

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    accelerator: str = "auto"
    strategy: str = "auto"
    devices: Union[list[int], str, int] = "auto"
    num_nodes: int = 1
    callbacks: list[Callback] = []
    precision: str = "32-true"
    logger: Optional[plLogger] = None
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = 10
    min_steps: Optional[int] = None
    limit_train_batches: int | float | None = None
    limit_val_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    limit_predict_batches: Optional[Union[int, float]] = None
    overfit_batches: Union[int, float] = 0.0
    val_check_interval: Optional[Union[int, float]] = None
    check_val_every_n_epoch: Optional[int] = 1
    num_sanity_val_steps: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    enable_checkpointing: Optional[bool] = None
    enable_progress_bar: Optional[bool] = None
    enable_model_summary: Optional[bool] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[Union[int, float]] = None
    gradient_clip_algorithm: Optional[str] = None
    default_root_dir: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return self.dict()


class TrainingConfigSchema(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True

    batch_size: int
    num_workers_for_dataloader: int = 8
    trainer: TrainerConfigSchema


class PretrainingModelAndDataset(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True

    model: BasePatientSlicePretrainer
    training_dataset: BasePatientSliceCollater
    validation_dataset: BasePatientSliceCollater


class ClassificationModelAndDataset(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True

    model: BasePatientSliceClassifier
    training_dataset: BasePredictionTimeCollater
    validation_dataset: BasePredictionTimeCollater


class ResolvedConfigSchema(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True

    # Required because dataset and model are coupled through their input and outputs
    model_and_dataset: PretrainingModelAndDataset | ClassificationModelAndDataset
    training: TrainingConfigSchema
    logger: Sequence[plLogger] | None = None
