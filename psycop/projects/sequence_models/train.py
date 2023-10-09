"""
- [x] Test that it runs
    - [x] Test that it saves a checkpoint
        - [x] Test that it can resume from a checkpoint
    - [ ] test that it run on gpu
- [ ] test that it logs to wandb and that we can upload it
    - [x] logs config (currently not logged)

TODO:
- [x] replace print with logging
- [x] fix moving to device
- [x] log hyperparameters
- [ ] log MLM accuracy

"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures.patient import Patient
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
    PatientLoader,
)
from psycop.common.sequence_models import PatientDataset
from psycop.common.sequence_models.checkpoint_savers.save_to_disk import (
    CheckpointToDisk,
)
from psycop.common.sequence_models.embedders import BEHRTEmbedder
from psycop.common.sequence_models.loggers.wandb_logger import WandbLogger
from psycop.common.sequence_models.tasks import BEHRTForMaskedLM
from psycop.common.sequence_models.trainer import Trainer

@dataclass
class ModelConfig:
    d_model: int = 32
    dropout_prob: float = 0.1
    max_sequence_length: int = 128
    nhead = int(d_model / 4)
    dim_feedforward = d_model * 4
    num_layers = 2


@dataclass
class TrainingConfig:
    project_name: str = "psycop-sequence-models"
    run_name: str = "initial-test"
    group: str = "testing"
    entity: str = "psycop"
    mode: str = "offline"
    device: str = "cuda"

    batch_size: int = 2
    n_steps: int = 100
    validate_every_n_steps: int = 1
    n_samples_to_validate_on: int = 2
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
        batch_first=True, 
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    # this includes the loss and the MLM head
    module = BEHRTForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
    )
    return module


config = Config()
model_cfg = config.model_config
training_cfg = config.training_config

train_patients = PatientLoader.get_split(
    event_loaders=[DiagnosisLoader()],
    split="train",
)
val_patients = PatientLoader.get_split(event_loaders=[DiagnosisLoader()], split="val")

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


def create_trainer(checkpoint_path: Path, config: TrainingConfig) -> Trainer:
    ckpt_saver = CheckpointToDisk(
        checkpoint_path=checkpoint_path,
        override_on_save=True,
    )

    logger = WandbLogger(
        project_name=config.project_name,
        run_name=config.run_name,
        group=config.group,
        entity=config.entity,
        mode=config.mode,
    )

    return Trainer(
        device=torch.device(config.device),
        validate_every_n_steps=config.validate_every_n_steps,
        n_samples_to_validate_on=config.n_samples_to_validate_on,
        logger=logger,
        checkpoint_savers=[ckpt_saver],
        save_every_n_steps=config.save_every_n_steps,
    )


project_root = Path(__file__).parents[4]
model_ckpt_path = project_root / "data" / "model_checkpoints"
model_ckpt_path.mkdir(parents=True, exist_ok=True)

trainer = create_trainer(checkpoint_path=model_ckpt_path, config=training_cfg)
trainer.logger.log_hyperparams(config)

trainer.fit(
    n_steps=training_cfg.n_steps,
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    resume_from_latest_checkpoint=False,
)
