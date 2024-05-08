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

def fit_from_config(sample_multiplier: int, cfg: Config):
    populate_baseline_registry()
    populate_scz_bp_registry()  # noqa: ERA001

    cfg_schema = BaselineSchema(**BaselineRegistry.resolve(cfg))
    cfg_schema.logger.log_config(cfg)
    cfg_schema.logger.log_metric(
        CalculatedMetric(name="sample_multiplier", value=sample_multiplier)
    )
    
    return cfg_schema.trainer.train().metric.value

def fit_ddpm_model(cfg: Config, sample_multiplier: int) -> float:
    cfg_copy = cfg.copy()
    cfg_copy["trainer"]["synthetic_data"]["n_samples"] = (
        base_positive_samples * sample_multiplier
    )

    result = fit_from_config(sample_multiplier=sample_multiplier, cfg=cfg_copy)
    return result

    
def fit_smote_model(cfg: Config, sample_multiplier: int) -> float:
    cfg_copy = cfg.copy()
    cfg_copy["trainer"]["task"]["task_pipe"]["sklearn_pipe"]["*"]["smote"]["n_minority_samples"] = (
        base_positive_samples * sample_multiplier
    )

    result = fit_from_config(sample_multiplier=sample_multiplier, cfg=cfg_copy)
    return result


if __name__ == "__main__":
    
    base_positive_samples = 15827
    
    # "scz_bp_structured_text_best_xgboost_synthcity.cfg", 
    cfgs = ["scz_bp_structured_text_best_xgboost_smote.cfg"]
    for cfg_path in cfgs:
        cfg_file = (
                Path(__file__).parent 
                / "config"
                / "augmentation"
                / cfg_path
            )

        cfg = Config().from_disk(cfg_file)
        fit_fn = fit_ddpm_model if "synthcity" in cfg_path else fit_smote_model

        fit_fn = partial(fit_fn, cfg=cfg)
        Parallel(n_jobs=10)(delayed(fit_fn)(sample_multiplier=sample_multiplier) for sample_multiplier in range(1, 11))
