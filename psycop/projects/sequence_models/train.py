"""
- [x] Test that it runs on Ovartaci
    - [ ] Test that it saves checkpoints at marked itervals
    - [ ] Test that it can resume from a checkpoint
    - [ ] Test that it runs on gpu
- [ ] Test that it logs to wandb and that we can upload it
    - [ ] Logs config (currently not logged)

TODO:
- [x] replace print with logging
- [x] fix moving to device
- [x] log hyperparameters
- [ ] log MLM accuracy

"""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures.patient import Patient
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
    PatientLoader,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_evaluation.utils import BaseModel
from psycop.common.sequence_models import PatientDataset
from psycop.common.sequence_models.embedders import BEHRTEmbedder
from psycop.common.sequence_models.tasks import BEHRTForMaskedLM


class ModelConfig(BaseModel):
    d_model: int = 32
    dropout_prob: float = 0.1
    max_sequence_length: int = 128
    nhead = int(d_model / 4)
    dim_feedforward = d_model * 4
    num_layers = 2


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

    batch_size: int = 2
    n_steps: int = 100
    validate_every_n_batches: int = 1
    save_every_n_steps: int = 1


@dataclass
class Config:
    training_config: TrainingConfig = TrainingConfig()  # noqa
    model_config: ModelConfig = ModelConfig()  # noqa

    def to_dict(self) -> dict[str, Any]:
        """return a flattened dictionary of the config"""

        d = self.training_config.__dict__
        d.update(self.model_config.__dict__)
        return d


def create_model(patients: list[Patient], config: ModelConfig) -> BEHRTForMaskedLM:
    """
    Creates a model for testing
    """
    emb = BEHRTEmbedder(
        d_model=config.d_model,
        dropout_prob=config.dropout_prob,
        max_sequence_length=config.max_sequence_length,
    )
    emb.fit(patients=patients, add_mask_token=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    # this includes the loss and the MLM head
    module = BEHRTForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
    )
    return module


def create_trainer(save_dir: Path, config: TrainingConfig) -> pl.Trainer:
    wandb_logger = pl_loggers.WandbLogger(
        name=config.run_name,
        save_dir=save_dir,
        offline=config.offline,
        project=config.project_name,
        log_model="all",
    )

    return pl.Trainer(
        accelerator=config.accelerator.value,
        max_steps=config.n_steps,
        val_check_interval=config.validate_every_n_batches,
        logger=wandb_logger,
    )


if __name__ == "__main__":
    config = Config()
    model_cfg = config.model_config
    training_cfg = config.training_config

    train_patients = PatientLoader.get_split(
        event_loaders=[DiagnosisLoader()],
        split="train",
    )
    val_patients = PatientLoader.get_split(
        event_loaders=[DiagnosisLoader()], split="val"
    )

    model = create_model(patients=train_patients, config=model_cfg)

    train_dataset = PatientDataset(train_patients)
    val_dataset = PatientDataset(val_patients)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
    )
    project_root = OVARTACI_SHARED_DIR / "sequence_models" / "BEHRT"
    project_root.mkdir(parents=True, exist_ok=True)

    save_dir = project_root / "data"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_trainer(save_dir=save_dir, config=training_cfg)
    trainer.logger.log_hyperparams(config.model_config.dict())

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
