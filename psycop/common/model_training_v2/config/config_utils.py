from pathlib import Path
from typing import Any

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import (
    BaselineSchema,
)
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)

populate_baseline_registry()

def load_baseline_config(config_path: Path) -> BaselineSchema:
    """Loads the baseline config from disk and resolves it."""
    cfg = Config().from_disk(config_path)

    # Fill with defaults and write to disk if relevant
    filled = BaselineRegistry.fill(cfg)
    if cfg != filled:
        filled.to_disk(config_path)
    
    resolved = BaselineRegistry.resolve(filled)
    return BaselineSchema(**resolved)


def load_hyperparam_config(config_path: Path) -> dict[str, Any]:
    """Loads the baseline config from disk and resolves it."""
    cfg = Config().from_disk(config_path)

    # Fill with defaults and write to disk if relevant
    filled = BaselineRegistry.fill(cfg)
    if cfg != filled:
        filled.to_disk(config_path)
    
    resolved = BaselineRegistry.resolve(filled)
    return resolved
