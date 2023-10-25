from collections.abc import Sequence
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from psycop.common.data_structures.patient import (
    PatientSlice,
)
from psycop.common.sequence_models import BEHRTForMaskedLM, PatientSliceDataset
from psycop.common.sequence_models.embedders.BEHRT_embedders import BEHRTEmbedder
from psycop.projects.sequence_models.pretrain import (
    Config,
    OptimizationConfig,
    TorchAccelerator,
    TrainingConfig,
    create_behrt_MLM_model,
    create_default_trainer,
)


def test_behrt(patient_dataset: PatientSliceDataset):
    d_model = 32
    emb = BEHRTEmbedder(d_model=d_model, dropout_prob=0.1, max_sequence_length=128)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=int(d_model / 4),
        dim_feedforward=d_model * 4,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    patients = patient_dataset.patient_slices
    emb.fit(patient_slices=patients, add_mask_token=True)

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
    patient_slices: Sequence[PatientSlice],
    tmp_path: Path,
):
    """
    Tests the general intended workflow of the Trainer class
    """

    n_patients = 10
    more_patients = list(patient_slices) * n_patients
    midpoint = int(n_patients / 2)

    train_patients = more_patients[:midpoint]
    val_patients = more_patients[midpoint:]

    train_dataset = PatientSliceDataset(train_patients)
    val_dataset = PatientSliceDataset(val_patients)

    config = Config(
        training_config=TrainingConfig(
            accelerator=TorchAccelerator.CPU,
            n_steps=midpoint,
            precision="32-true",
        ),
        optimization_config=OptimizationConfig(
            lr_scheduler_kwargs={"num_warmup_steps": 2, "num_training_steps": midpoint},
        ),
    )

    trainable_module = create_behrt_MLM_model(
        patient_slices=train_patients,
        config=config,
    )

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
