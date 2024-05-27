from pathlib import Path

import confection

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_cvd_registry()

    cfg = confection.Config().from_disk(Path(__file__).parent / "cvd_baseline.cfg")

    cfg["trainer"]["task"]["task_pipe"]["sklearn_pipe"]["*"]["model"] = {
        "@estimator_steps_suggesters": "xgboost_suggester"
    }

    # Set run name
    cfg["logger"]["*"]["mlflow"]["run_name"] = "CVD hyperparam tuning, xgboost"
    cfg["trainer"]["preprocessing_pipeline"]["*"]["layer_selector"][
        "keep_matching"
    ] = ".+_layer_(1|2).+"

    OptunaHyperParameterOptimization().from_cfg(
        cfg,
        study_name="cvd_hyperparam_tuning",
        n_trials=150,
        n_jobs=30,
        direction="maximize",
        catch=(Exception,),
        custom_populate_registry_fn=populate_with_cvd_registry,
    )
