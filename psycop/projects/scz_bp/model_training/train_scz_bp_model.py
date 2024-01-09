from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_pipeline import (
    train_baseline_model_from_schema,
)
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import (
    populate_scz_bp_registry,
)
from psycop.projects.scz_bp.model_training.scz_bp_registry import SczBpRegistry


def train_scz_bp_baseline_model(cfg_file: Path) -> float:
    cfg = Config().from_disk(cfg_file)
    cfg_schema = BaselineSchema(**SczBpRegistry.resolve(cfg))

    cfg_schema.logger.log_config(
        cfg,
    )
    cfg_schema.logger.warn(
        """Config is not filled, so defaults will not be logged.
                           Waiting for https://github.com/explosion/confection/issues/47 to be resolved.""",
    )

    return train_baseline_model_from_schema(cfg_schema)


if __name__ == "__main__":
    populate_baseline_registry()
    populate_scz_bp_registry()
    train_scz_bp_baseline_model(
        Path(__file__).parent / "config" / "scz_bp_baseline.cfg",
    )
