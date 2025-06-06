import copy
import logging
from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry


def train_cvd_layers(cfg: PsycopConfig):
    for layer in range(1, 5):
        layer_cfg = copy.deepcopy(cfg)
        layer_cfg.mut("logger.*.mlflow.run_name", f"CVD layer {layer}, base")

        layers = [str(i) for i in range(1, layer + 1)]
        layer_cfg.mut(
            "trainer.preprocessing_pipeline.*",
            {"layer_selector": {"keep_matching": f".+_layer_({'|'.join(layers)}).+"}},
        )

        logging.info(f"Training model with layers {layers}")
        train_baseline_model_from_cfg(cfg=layer_cfg)

        if layer == 1:
            aggs = ".+(mean|min|max).+"
            layer_cfg.add(
                "trainer.preprocessing_pipeline.*.aggregation_selector.keep_matching", aggs
            )
            layer_cfg.mut("logger.*.mlflow.run_name", f"CVD layer {layer}, (mean, min, mx)")

            logging.info(f"Training model with {aggs}")
            train_baseline_model_from_cfg(cfg=layer_cfg)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    populate_baseline_registry()
    populate_with_cvd_registry()

    train_cvd_layers(cfg=PsycopConfig().from_disk(Path(__file__).parent / "cvd_baseline.cfg"))
