from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_cvd_registry()
    OptunaHyperParameterOptimization().from_file(
        (Path(__file__).parent / "cvd_baseline.cfg"),
        study_name="test_joblib",
        n_trials=1000,
        n_jobs=30,
        direction="maximize",
        catch=(Exception,),
    )
