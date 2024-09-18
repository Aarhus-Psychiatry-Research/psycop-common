from pathlib import Path

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.t2d_bigdata.model_training.populate_t2d_bigdata_registry import populate_with_t2d_bigdata_registry


def logistic_regression_hyperparam(cfg: PsycopConfig):
    cfg.mut(
        "trainer.task.task_pipe.sklearn_pipe.*.model",
        {
            "@estimator_steps_suggesters": "logistic_regression_suggester",
            "penalties": ("elasticnet",),
        },
    )

    # Add the standard_scaler and imputer
    pipe_items = [
        ("imputer", {"@estimator_steps": "simple_imputation", "strategy": "median"}),
        ("scaler", {"@estimator_steps": "standard_scaler"}),
        *cfg.retrieve("trainer.task.task_pipe.sklearn_pipe.*").items(),
    ]
    cfg.mut("trainer.task.task_pipe.sklearn_pipe.*", dict(pipe_items))

    # Set run name
    for i in reversed([1, 2, 3, 4]):
        cfg.mut("logger.*.mlflow.experiment_name", f"CVD, h, l-{i}, logR")

        # Filter by layers
        layer_regex = "|".join([str(i) for i in range(1, i + 1)])
        cfg.mut(
            "trainer.preprocessing_pipeline.*.layer_selector.keep_matching",
            f".+_layer_({layer_regex}).+",
        )

        OptunaHyperParameterOptimization().from_cfg(
            cfg,
            study_name=cfg.retrieve("logger.*.mlflow.experiment_name") + "2",
            n_trials=100,
            n_jobs=30,
            direction="maximize",
            catch=(Exception,),
            custom_populate_registry_fn=populate_with_t2d_bigdata_registry,
        )


if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_t2d_bigdata_registry()

    logistic_regression_hyperparam(
        PsycopConfig().from_disk(Path(__file__).parent / "cvd_baseline.cfg")
    )
