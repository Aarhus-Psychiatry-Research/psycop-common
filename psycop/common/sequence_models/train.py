"""
The main training entrypoint for sequence models.
"""
import logging
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .config_utils import flatten_nested_dict, load_config, parse_config

std_logger = logging.getLogger(__name__)


def populate_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    from .embedders.BEHRT_embedders import create_behrt_embedder
    from .logger import create_wandb_logger
    from .model_layers import create_encoder_layer, create_transformers_encoder
    from .optimizers import (
        create_adam,
        create_adamw,
        create_linear_schedule_with_warmup,
    )
    from .tasks import create_behrt, create_encoder_for_clf


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
    flat_config = flatten_nested_dict(config_dict)
    logger.experiment.config.update(flat_config)

    # create dataloader:
    train_loader = DataLoader(
        training_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
        num_workers=training_cfg.num_workers_for_dataloader,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=model.collate_fn,
        num_workers=training_cfg.num_workers_for_dataloader,
    )

    trainer = pl.Trainer(**trainer_kwargs)

    std_logger.info("Starting training")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)  # type: ignore