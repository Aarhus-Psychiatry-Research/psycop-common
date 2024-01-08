"""
The main training entrypoint for sequence models.
"""
import logging
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from psycop.common.global_utils.config_utils import (
    flatten_nested_dict,
)

from ..model_training_v2.loggers.mlflow_logger import sanitise_dict_keys
from .config_utils import load_config, parse_config

log = logging.getLogger(__name__)
os.environ["WANDB__SERVICE_WAIT"] = "300"  # to avoid issues with wandb service


def populate_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    from .callbacks import create_learning_rate_monitor, create_model_checkpoint  # noqa
    from .embedders.BEHRT_embedders import create_behrt_embedder  # noqa
    from .logger import create_mlflow_logger, create_wandb_logger  # noqa
    from .model_layers import create_encoder_layer, create_transformers_encoder  # noqa
    from .optimizers import create_adam  # noqa
    from .optimizers import create_adamw  # noqa
    from .optimizers import create_linear_schedule_with_warmup  # noqa
    from .tasks.pretrainer_behrt import PretrainerBEHRT  # noqa


populate_registry()


def train(config_path: Path | None = None) -> None:
    """
    Train a model based on the config

    Args:
        config_path: path to config file if none loads default
    """
    config_dict = load_config(config_path)
    cfg = parse_config(config_dict)

    if cfg.logger is not None:
        for logger in cfg.logger:
            # update config
            log.info("Logging config")
            flat_config = flatten_nested_dict(config_dict)
            logger.log_hyperparams(sanitise_dict_keys(flat_config))

    # Load and filter dataset
    filter_fn = cfg.model_and_dataset.model.filter_and_reformat

    log.info("Preparing train")
    training_dataset = cfg.model_and_dataset.training_dataset.get_dataset()
    training_dataset.filter_patients(filter_fn)

    log.info("Preparing validation")
    validation_dataset = cfg.model_and_dataset.validation_dataset.get_dataset()
    validation_dataset.filter_patients(filter_fn)

    log.info("Fitting embedder")
    embedder = cfg.model_and_dataset.model.embedder
    if not embedder.is_fitted:
        embedder.fit(training_dataset.patient_slices)
    else:
        log.info("Embedder already fitted, continuing")

    log.info("Creating dataloaders")
    train_loader = DataLoader(
        training_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=cfg.model_and_dataset.model.collate_fn,
        num_workers=cfg.training.num_workers_for_dataloader,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=cfg.model_and_dataset.model.collate_fn,
        num_workers=cfg.training.num_workers_for_dataloader,
        persistent_workers=True,
    )

    log.info("Initalizing trainer")
    trainer = pl.Trainer(**cfg.training.trainer.to_dict())

    log.info("Starting training")
    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        model=cfg.model_and_dataset.model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
