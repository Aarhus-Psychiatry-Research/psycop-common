import logging
from pathlib import Path

import confection

from psycop.common.model_training_v2.config.baseline_pipeline import (
    train_baseline_model,
    train_baseline_model_from_schema,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry

if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    populate_baseline_registry()
    populate_with_cvd_registry()

    cfg = confection.Config().from_disk(Path(__file__).parent / "cvd_baseline.cfg")
    # TODO: Change trainer.training_data paths to lookbehind_experiments
    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg))
    train_baseline_model_from_schema(cfg_schema)
