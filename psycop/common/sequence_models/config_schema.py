"""
The config Schema for sequence models.
"""

from pathlib import Path
from typing import Any, Optional, Union

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from pydantic import BaseModel
from torch.utils.data import Dataset

from psycop.common.data_structures import PatientSlice
from psycop.common.sequence_models.tasks import (
    BEHRTForMaskedLM,
    EncoderForClassification,
)


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
    logger: WandbLogger
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
    batch_size: int
    num_workers_for_dataloader: int = 8
    trainer: TrainerConfigSchema


class DatasetsConfigSchema(BaseModel):
    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    training: Dataset[PatientSlice]
    validation: Dataset[PatientSlice]


class ResolvedConfigSchema(BaseModel):
    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    dataset: DatasetsConfigSchema
    model: BEHRTForMaskedLM | EncoderForClassification
    training: TrainingConfigSchema
