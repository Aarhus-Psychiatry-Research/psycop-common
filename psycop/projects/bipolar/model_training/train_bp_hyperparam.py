from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.bipolar.model_training.populate_bp_registry import populate_with_bp_registry

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_bp_registry()
    OptunaHyperParameterOptimization().from_file(
        (Path(__file__).parent / "bp_text_model.cfg"),
        study_name="bipolar_model_training_text_feature_lb_200_interval_150",
        n_trials=150,
        n_jobs=10,
        direction="maximize",
        catch=(Exception,),
    )
