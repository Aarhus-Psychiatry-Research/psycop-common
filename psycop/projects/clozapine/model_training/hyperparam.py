from pathlib import Path

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)

FEATURE_SETS = {
    "structured_text": [
        "contacts",
        "selvmord-broset",
        "diagnoses",
        "medication",
        "unique_count_medication",
        "depot-medication",
        "lab-results",
        "coercion",
        "ect",
        "text",
    ]
}


def hyperparameter_search(cfg: PsycopConfig):
    cfg.mut(
        "trainer.task.task_pipe.sklearn_pipe.*.model",
        {"@estimator_steps_suggesters": "xgboost_suggester"},
    )

    # Set run name
    for feature_set, features in FEATURE_SETS.items():
        cfg.mut(
            "logger.*.mlflow.experiment_name",
            f"clozapine hparam, {feature_set}, xgboost, no lookbehind filter",
        )

        layer_regex = "|".join(features)

        cfg.mut(
            "trainer.preprocessing_pipeline.*.layer_selector.keep_matching",
            f".+_layer_({layer_regex}).+",
        )

        OptunaHyperParameterOptimization().from_cfg(
            cfg,
            study_name=cfg.retrieve("logger.*.mlflow.experiment_name"),
            n_trials=150,
            n_jobs=10,
            direction="maximize",
            catch=(Exception,),
            custom_populate_registry_fn=None,
        )


if __name__ == "__main__":
    populate_baseline_registry()

    hyperparameter_search(
        PsycopConfig().from_disk(Path(__file__).parent / "clozapine_baseline.cfg")
    )
