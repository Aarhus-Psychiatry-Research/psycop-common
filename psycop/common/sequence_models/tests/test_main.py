from pathlib import Path

import pytest
from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures import Patient
from psycop.common.sequence_models.embedders.BEHRT_embedders import BEHRTEmbedder
from psycop.common.sequence_models import (
    BEHRTForMaskedLM,
    PatientDataset,
)
from psycop.projects.sequence_models.train import (
    Config,
    OptimizationConfig,
    TorchAccelerator,
    TrainingConfig,
    create_behrt_MLM_model,
    create_default_trainer,
)


@pytest.fixture()
def patient_dataset(patients: list) -> PatientDataset:
    return PatientDataset(patients)


def test_behrt(patient_dataset: PatientDataset):
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    patients = patient_dataset.patients
    emb.fit(patients, add_mask_token=True)

    config = Config()

    behrt = BEHRTForMaskedLM(
        embedding_module=emb,
        encoder_module=encoder,
        optimizer_kwargs=config.optimization_config.optimizer_kwargs,
        lr_scheduler_kwargs=config.optimization_config.lr_scheduler_kwargs,
    )

    dataloader = DataLoader(
        patient_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=behrt.collate_fn,
    )

    for input_ids, masked_labels in dataloader:
        output = behrt(input_ids, masked_labels)
        loss = output["loss"]
        loss.backward()  # ensure that the backward pass works


def test_module_with_trainer(
    patients: list[Patient],
    tmp_path: Path,
):
    """
    Tests the general intended workflow of the Trainer class
    """

    n_patients = 10
    patients = patients * n_patients
    midpoint = int(n_patients / 2)

    train_patients = patients[:midpoint]
    val_patients = patients[midpoint:]

    train_dataset = PatientDataset(train_patients)
    val_dataset = PatientDataset(val_patients)

    config = Config(
        training_config=TrainingConfig(
            accelerator=TorchAccelerator.CPU,
            n_steps=midpoint,
        ),
        optimization_config=OptimizationConfig(
            lr_scheduler_kwargs={"num_warmup_steps": 2, "num_training_steps": midpoint},
        ),
    )

    trainable_module = create_behrt_MLM_model(patients=train_patients, config=config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True,
        collate_fn=trainable_module.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True,
        collate_fn=trainable_module.collate_fn,
    )

    trainer = create_default_trainer(save_dir=tmp_path, config=config)
    trainer.fit(
        model=trainable_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Checkpoints are saved
    checkpoint_paths = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(checkpoint_paths) >= 1

    # Checkpoint can be loaded
    # Note that load_from_checkpoint raises a FileNotFoundError if the checkpoint does not exist.
    # Hence, this would fail if we could not load the checkpoint.
    loaded_model = BEHRTForMaskedLM.load_from_checkpoint(checkpoint_paths[0])
    trainer.fit(model=loaded_model, train_dataloaders=train_dataloader)
