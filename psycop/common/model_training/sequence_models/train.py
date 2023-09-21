"""
- [x] Test that it runs
    - [x] Test that it saves a checkpoint
        - [x] Test that it can resume from a checkpoint
        - [ ] check that it does not overwrite the new device
    - [ ] test that it run on gpu
- [ ] test that it logs to wandb
    - [ ] logs config (currently not logged)

TODO:
- replace print with logging
- fix moving to device
- log hyperparameters
"""


from pathlib import Path
import torch

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
from torch import nn

from psycop.common.sequence_models.trainer import Trainer



def create_model(patients: list[Patient]) -> BEHRTForMaskedLM:
    """
    Creates a model for testing
    """
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    emb.fit(patients=patients, add_mask_token=True)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    # this includes the loss and the MLM head
    module = BEHRTForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
    )
    return module


train_patients = PatientLoader.get_split(
    event_loaders=[DiagnosisLoader()], split="train"
)
val_patients = PatientLoader.get_split(event_loaders=[DiagnosisLoader()], split="val")

model = create_model(patients=train_patients)

train_dataset = PatientDataset(train_patients)
val_dataset = PatientDataset(val_patients)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=model.collate_fn,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=model.collate_fn,
)


def create_trainer(checkpoint_path: Path) -> Trainer:
    ckpt_saver = CheckpointToDisk(
        checkpoint_path=checkpoint_path,
        override_on_save=True,
    )

    logger = WandbLogger(
        project_name="psycop-sequence-models",
        run_name="initial-test",
        group="testing",
        entity="psycop",
        mode="offline",
    )

    return Trainer(
        device=torch.device("cuda"),
        validate_every_n_steps=1,
        n_samples_to_validate_on=2,
        logger=logger,
        checkpoint_savers=[ckpt_saver],
        save_every_n_steps=1,
    )


project_root = Path(__file__).parents[4]
model_ckpt_path = project_root / "data" / "model_checkpoints"
model_ckpt_path.mkdir(parents=True, exist_ok=True)

trainer = create_trainer(checkpoint_path=model_ckpt_path)
trainer.fit(
    n_steps=100,
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    resume_from_latest_checkpoint=False,
)
