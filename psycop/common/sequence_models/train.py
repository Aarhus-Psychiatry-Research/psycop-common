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
    config = parse_config(config_dict)

    training_cfg = config.training
    training_dataset = config.dataset.training
    validation_dataset = config.dataset.validation
    model = config.model
    logger = training_cfg.trainer.logger
    trainer_kwargs = training_cfg.trainer.to_dict()

    # update config
    std_logger.info("Updating Config")
    flat_config = flatten_nested_dict(config_dict)
    logger.experiment.config.update(flat_config)

    # filter dataset
    std_logger.info("Filtering Patients")
    filter_fn = model.embedding_module.A_diagnoses_to_caliber
    training_dataset.filter_patients(filter_fn)
    validation_dataset.filter_patients(filter_fn)

    std_logger.info("Creating dataloaders")
    train_loader = DataLoader(
        training_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
        num_workers=training_cfg.num_workers_for_dataloader,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        collate_fn=model.collate_fn,
        num_workers=training_cfg.num_workers_for_dataloader,
        persistent_workers=True,
    )

    std_logger.info("Initalizing trainer")
    trainer = pl.Trainer(**trainer_kwargs)

    std_logger.info("Starting training")
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
