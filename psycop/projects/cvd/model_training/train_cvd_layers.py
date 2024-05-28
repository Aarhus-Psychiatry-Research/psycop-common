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

    for layer in range(1, 5):
        cfg = confection.Config().from_disk(Path(__file__).parent / "cvd_baseline.cfg")
        layers = [str(i) for i in range(1, layer + 1)]
        cfg["trainer"]["preprocessing_pipeline"]["*"]["layer_selector"]["keep_matching"] = (
            f".+_layer_({'|'.join(layers)}).+"
        )

        logging.info(f"Training model with layers {layers}")
        train_baseline_model_from_cfg(cfg=cfg)

        if layer == 1:
            aggs = ".+(mean|min|max).+"
            cfg["trainer"]["preprocessing_pipeline"]["*"]["aggregation_selector"][
                "keep_matching"
            ] = aggs

            logging.info(f"Training model with {aggs}")
            train_baseline_model_from_cfg(cfg=cfg)
