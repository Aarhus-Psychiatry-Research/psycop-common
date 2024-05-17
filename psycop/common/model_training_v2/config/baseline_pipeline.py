from pathlib import Path

import confection
from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema


def train_baseline_model(cfg_file: Path) -> float:
    """When you just want to use a file."""
    cfg = Config().from_disk(cfg_file)
    return train_baseline_model_from_cfg(cfg)


def train_baseline_model_from_cfg(cfg: confection.Config) -> float:
    """When you want to programatically change options before training, while logging the cfg."""
    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg))
    cfg_schema.logger.log_config(cfg)
    cfg_schema.logger.warn(
        """Config is not filled, so defaults will not be logged.
                           Waiting for https://github.com/explosion/confection/issues/47 to be resolved."""
    )

    return train_baseline_model_from_schema(cfg_schema)


def train_baseline_model_from_schema(cfg: BaselineSchema) -> float:
    """For just training"""
    result = cfg.trainer.train()

    return result.metric.value
