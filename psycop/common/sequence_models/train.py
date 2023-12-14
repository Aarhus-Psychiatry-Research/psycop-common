"""
The main training entrypoint for sequence models.
"""
import logging
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from psycop.common.global_utils.config_utils import flatten_nested_dict

from .config_utils import load_config, parse_config

std_logger = logging.getLogger(__name__)
os.environ["WANDB__SERVICE_WAIT"] = "300"  # to avoid issues with wandb service


def populate_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    from .callbacks import create_learning_rate_monitor, create_model_checkpoint  # noqa
    from .embedders.BEHRT_embedders import create_behrt_embedder  # noqa
    from .logger import create_wandb_logger  # noqa
    from .model_layers import create_encoder_layer, create_transformers_encoder  # noqa
    from .optimizers import create_adam  # noqa
    from .optimizers import create_adamw  # noqa
    from .optimizers import create_linear_schedule_with_warmup  # noqa
    from .tasks import create_behrt, create_encoder_for_clf  # noqa


populate_registry()


def train(config_path: Path | None = None) -> None:
    """
    Train a model based on the config

    Args:
        config_path: path to config file if none loads default
    """
    config_dict = load_config(config_path)
    cfg = parse_config(config_dict)

    # Config
    std_logger.info("Updating Config")
    cfg.training.trainer.logger.experiment.config.update(
        flatten_nested_dict(config_dict),
    )

    # Dataset
    std_logger.info("Filtering Patients")
    filter_fn = cfg.model.embedding_module.A_diagnoses_to_caliber
    cfg.dataset.training.filter_patient_slices(filter_fn)
    cfg.dataset.validation.filter_patient_slices(filter_fn)

    # Dataloaders
    std_logger.info("Creating dataloaders")
    train_loader = DataLoader(
        cfg.dataset.training,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=cfg.model.collate_fn,
        num_workers=cfg.training.num_workers_for_dataloader,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        cfg.dataset.validation,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=cfg.model.collate_fn,
        num_workers=cfg.training.num_workers_for_dataloader,
        persistent_workers=True,
    )

    # Trainer and training
    std_logger.info("Initalizing trainer")
    trainer = pl.Trainer(**cfg.training.trainer.to_dict())

    std_logger.info("Starting training")
    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        model=cfg.model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
