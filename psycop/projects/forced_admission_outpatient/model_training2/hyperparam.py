# Script for running hyperparametertuning from baseline configuration
from pathlib import Path

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)


def hyperparameter_search(cfg: PsycopConfig):
    cfg.mut(
        "trainer.task.task_pipe.sklearn_pipe.*.model",
        {"@estimator_steps_suggesters": "xgboost_suggester"},
    )

    cfg.mut("logger.*.mlflow.experiment_name", "uti_hparam_test_run")

    cfg.mut(
        "logger.*.disk_logger.run_path",
        "E:/shared_resources/ect/model_training/uti_hparam_test_run",
    )

    OptunaHyperParameterOptimization().from_cfg(
        cfg,
        study_name=cfg.retrieve("logger.*.mlflow.experiment_name") + "_",
        n_trials=100,
        n_jobs=10,
        direction="maximize",
        catch=(Exception,),
        custom_populate_registry_fn=None,
    )


if __name__ == "__main__":
    populate_baseline_registry()

    hyperparameter_search(PsycopConfig().from_disk(Path(__file__).parent / "fao_baseline.cfg"))
