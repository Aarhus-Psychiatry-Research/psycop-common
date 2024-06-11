from pathlib import Path
from typing import Any

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

populate_baseline_registry()

def resolve_and_fill_config(config_path: Path, fill_cfg_with_defaults: bool) -> dict[str, Any]:
    cfg = Config().from_disk(config_path)

    # Fill with defaults
    if fill_cfg_with_defaults:
        filled = BaselineRegistry.fill(cfg, validate=False)
        resolved = BaselineRegistry.resolve(filled)
        # Writing to disk happens after resolution, to ensure that the
        # config is valid
        if cfg != filled:
            filled.to_disk(config_path)
    else:
        resolved = BaselineRegistry.resolve(cfg)

    return resolved


def load_baseline_config(config_path: Path) -> BaselineSchema:
    """Loads the baseline config from disk and resolves it."""
    resolved = resolve_and_fill_config(config_path, fill_cfg_with_defaults=True)
    return BaselineSchema(**resolved)
