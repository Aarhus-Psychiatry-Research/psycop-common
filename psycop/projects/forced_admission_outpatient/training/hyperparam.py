# Script for running hyperparametertuning from baseline configuration
from pathlib import Path

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.forced_admission_outpatient.training.populate_fao_registry import (
    populate_with_fao_registry,
)


def hyperparameter_search(cfg: PsycopConfig, run_name: str):
    cfg.mut("logger.*.mlflow.experiment_name", f"{run_name}")

    cfg.mut(
        "logger.*.disk_logger.run_path",
        f"E:/shared_resources/forced_admissions_outpatient/model_training/{run_name}",
    )

    OptunaHyperParameterOptimization().from_cfg(
        cfg,
        study_name=cfg.retrieve("logger.*.mlflow.experiment_name") + "_",
        n_trials=100,
        n_jobs=5,
        direction="maximize",
        catch=(Exception,),
        custom_populate_registry_fn=None,
    )


if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_fao_registry()
    hyperparameter_search(
        PsycopConfig().from_disk(Path(__file__).parent / "fao_all_features_hyperparam.cfg"),
        run_name="full_model_all_features",
    )
