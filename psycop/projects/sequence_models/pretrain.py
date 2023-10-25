"""
- [x] Test that it runs on Ovartaci
    - [x] Test that it saves checkpoints at marked itervals
    - [x] Test that it can resume from a checkpoint
    - [x] Test that it runs on gpu
- [x] Test that it logs to wandb and that we can upload it
    - [x] Logs config (currently not logged)

TODO:
- [x] replace print with logging
- [x] fix moving to device
- [x] log hyperparameters
- [ ] log MLM accuracy

"""

import enum
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures.patient import (
    PatientSlice,
)
from psycop.common.feature_generation.loaders.raw.load_ids import SplitName
from psycop.common.feature_generation.sequences.event_dataframes_to_patient import (
    PatientSliceColumnNames,
)
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
    PatientLoader,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.sequence_models import PatientSliceDataset
from psycop.common.sequence_models.embedders.BEHRT_embedders import BEHRTEmbedder
from psycop.common.sequence_models.tasks import BEHRTForMaskedLM


@dataclass
class ModelConfig:
    d_model: int = 288
    num_layers = 6
    n_heads = 12
    dim_feedforward = 512
    dropout_prob: float = 0.1
    max_sequence_length: int = 256
    map_diagnosis_codes: bool = True


class TorchAccelerator(enum.Enum):
    CPU = "cpu"
    METAL = "mps"
    CUDA = "cuda"


@dataclass
class TrainingConfig:
    project_name: str = "psycop-sequence-models"
    run_name: str = "initial-test"
    group: str = "testing"
    entity: str = "psycop"
    offline: bool = True
    accelerator: TorchAccelerator = TorchAccelerator.CUDA

    n_steps: int = 100_000
    batch_size: int = 512
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"
    validate_every_prop_epoch: float = 1.0
    checkpoint_every_n_epochs: int = 1

    # data filtering
    min_n_visits: int = 5
    patient_column_names: PatientSliceColumnNames | None = field(
        default=PatientSliceColumnNames(
            source_subtype_col_name="type",
        ),
    )


@dataclass
class OptimizationConfig:
    optimizer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-3,
        },
    )
    lr_scheduler_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "num_warmup_steps": 10_000,
            "num_training_steps": TrainingConfig.n_steps,
        },
    )


@dataclass
class Config:
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)

    def to_dict(self) -> dict[str, Any]:
        """return a flattened dictionary of the config"""

        d = self.training_config.__dict__
        d.update(self.model_config.__dict__)
        d.update(self.optimization_config.__dict__)
        return d


def create_behrt_MLM_model(
    patient_slices: Sequence[PatientSlice],
    config: Config,
) -> BEHRTForMaskedLM:
    """
    Creates a model for testing
    """
    emb = BEHRTEmbedder(
        d_model=config.model_config.d_model,
        dropout_prob=config.model_config.dropout_prob,
        max_sequence_length=config.model_config.max_sequence_length,
    )
    emb.fit(
        patient_slices=patient_slices,
        add_mask_token=True,
        map_diagnosis_codes=config.model_config.map_diagnosis_codes,
    )

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.model_config.d_model,
        nhead=config.model_config.n_heads,
        dim_feedforward=config.model_config.dim_feedforward,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=config.model_config.num_layers,
    )

    # this includes the loss and the MLM head
    module = BEHRTForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
        optimizer_kwargs=config.optimization_config.optimizer_kwargs,
        lr_scheduler_kwargs=config.optimization_config.lr_scheduler_kwargs,
    )
    return module


def create_default_trainer(save_dir: Path, config: Config) -> pl.Trainer:
    wandb_logger = pl_loggers.WandbLogger(
        name=config.training_config.run_name,
        save_dir=save_dir,
        offline=config.training_config.offline,
        project=config.training_config.project_name,
    )

    trainer = pl.Trainer(
        accelerator=config.training_config.accelerator.value,
        val_check_interval=config.training_config.validate_every_prop_epoch,
        logger=wandb_logger,
        max_steps=config.training_config.n_steps,
        accumulate_grad_batches=config.training_config.accumulate_grad_batches,
        precision=config.training_config.precision,  # type: ignore
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir / "checkpoints",
                every_n_epochs=config.training_config.checkpoint_every_n_epochs,
                verbose=True,
                save_top_k=5,
                mode="min",
                monitor="val_loss",
            ),
            LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        ],
    )
    wandb_logger.experiment.config.update(asdict(config))

    return trainer


if __name__ == "__main__":
    run_name = input("Enter a run name: ")

    config = Config(
        training_config=TrainingConfig(
            accelerator=TorchAccelerator.CUDA,
            run_name=run_name,
        ),
    )

    train_patients = PatientLoader.get_split(
        event_loaders=[
            DiagnosisLoader(min_n_visits=config.training_config.min_n_visits),
        ],
        split=SplitName.TRAIN,
        patient_column_names=config.training_config.patient_column_names,
    )
    val_patients = PatientLoader.get_split(
        event_loaders=[
            DiagnosisLoader(min_n_visits=config.training_config.min_n_visits),
        ],
        split=SplitName.VALIDATION,
        patient_column_names=config.training_config.patient_column_names,
    )
    train_dataset = PatientSliceDataset([p.as_slice() for p in train_patients])
    val_dataset = PatientSliceDataset([p.as_slice() for p in val_patients])

    model = create_behrt_MLM_model(
        patient_slices=[p.as_slice() for p in train_patients],
        config=config,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
    )
    project_root = OVARTACI_SHARED_DIR / "sequence_models" / "BEHRT"
    project_root.mkdir(parents=True, exist_ok=True)

    save_dir = project_root / "data"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_default_trainer(save_dir=save_dir, config=config)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
