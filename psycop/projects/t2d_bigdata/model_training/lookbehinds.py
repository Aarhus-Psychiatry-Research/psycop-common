import logging
from collections.abc import Sequence
from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.t2d_bigdata.model_training.populate_t2d_bigdata_registry import (
    populate_with_t2d_bigdata_registry,
)


def train_with_lookbehinds(cfg: PsycopConfig, lookbehinds: Sequence[int]):
    for i, _ in enumerate(lookbehinds):
        distances_i = [str(lookbehind) for lookbehind in lookbehinds[-i - 1 :]]
        distances = "|".join(distances_i)

        cfg.mut(
            "trainer.preprocessing_pipeline.*.lookbehind_selector.keep_matching",
            f".+_({distances})_.+",
        )
        cfg.mut(
            "logger.*.mlflow.run_name", f"T2D-bigdata layer 1, lookbehind: {','.join(distances_i)}"
        )

        logging.info(f"Training model with {distances_i}")
        train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    populate_baseline_registry()
    populate_with_t2d_bigdata_registry()

    cfg = PsycopConfig().from_disk(Path(__file__).parent / "t2d_bigdata_baseline.cfg")

    lookbehinds = [90, 365, 730]
    train_with_lookbehinds(cfg, lookbehinds)
