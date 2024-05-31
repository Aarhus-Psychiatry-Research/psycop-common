import logging
from pathlib import Path

import confection

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
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

    lookbehinds = [90, 365, 730]
    for i, _ in enumerate(lookbehinds):
        distances_i = [str(lookbehind) for lookbehind in lookbehinds[-i - 1 :]]
        distances = "|".join(distances_i)

        cfg["trainer"]["preprocessing_pipeline"]["*"]["lookbehind_selector"]["keep_matching"] = (
            f".+_({distances})_.+"
        )
        cfg["logger"]["*"]["mlflow"]["run_name"] = (
            f"CVD layer 1, lookbehind: {','.join(distances_i)}"
        )

        logging.info(f"Training model with {distances_i}")
        train_baseline_model_from_cfg(cfg)
