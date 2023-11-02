"""
The main config for sequence models.
"""


from pathlib import Path

from confection import Config

from psycop.common.sequence_models import Registry
from psycop.common.sequence_models.config_schema import ResolvedConfigSchema

default_config_path = Path(__file__).parent / "default_config.cfg"


def load_config(config_path: Path | None = None) -> Config:
    if config_path is None:
        config_path = default_config_path
    cfg = Config().from_disk(config_path)
    return cfg


def parse_config(config: Config) -> ResolvedConfigSchema:
    cfg = Registry.resolve(config)
    return ResolvedConfigSchema(**cfg)
