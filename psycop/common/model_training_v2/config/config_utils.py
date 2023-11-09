
from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import (
    BaselineSchema,
)


def load_config(config_path: Path) -> Config:
    cfg = Config().from_disk(config_path)
    return cfg


def resolve_config(config: Config) -> BaselineSchema:
    """Gets a config object and resolves it from the registry"""
    resolved= BaselineRegistry.resolve(config)
    return BaselineSchema(**resolved)  


def load_baseline_config(config_path: Path) -> BaselineSchema:
    """Loads the baseline config from disk and resolves it."""
    cfg = Config().from_disk(config_path)
    return resolve_config(cfg)