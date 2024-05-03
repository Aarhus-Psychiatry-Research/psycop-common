from functools import partial
from pathlib import Path

from confection import Config

from joblib import Parallel, delayed
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric

from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import (
    populate_scz_bp_registry,
)


def fit_ddpm_model(cfg: Config, sample_multiplier: int) -> None:
    populate_baseline_registry()
    populate_scz_bp_registry()  # noqa: ERA001

    cfg_copy = cfg.copy()
    cfg_copy["trainer"]["synthetic_data"]["n_samples"] = (
        base_positive_samples * sample_multiplier
    )

    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg_copy))
    cfg_schema.logger.log_config(cfg_copy)
    cfg_schema.logger.log_metric(
        CalculatedMetric(name="sample_multiplier", value=sample_multiplier)
    )
    
    _ = cfg_schema.trainer.train()
    

if __name__ == "__main__":
    cfg_file = (
        Path(__file__).parent
        / "config"
        / "augmentation"
        / "scz_bp_structured_text_best_xgboost_synthcity.cfg"
    )
    # or "scz_bp_structured_text_best_xgboost_smote.cfg" for smote
    cfg = Config().from_disk(cfg_file)

    base_positive_samples = 14914

    fit_fn = partial(fit_ddpm_model, cfg=cfg)
    Parallel(n_jobs=10)(delayed(fit_fn)(sample_multiplier=sample_multiplier) for sample_multiplier in range(1, 11))
