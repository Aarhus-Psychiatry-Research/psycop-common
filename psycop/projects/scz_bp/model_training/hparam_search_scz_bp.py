from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

if __name__ == "__main__":
    populate_baseline_registry()
    populate_scz_bp_registry()
    OptunaHyperParameterOptimization().from_file(
        (
            Path(__file__).parent
            / "config"
            / "main"
            / "scz_bp_structured_text_xgboost.cfg"
        ),
        study_name="test_feature_gen",
        n_trials=11,
        n_jobs=1,
        direction="maximize",
        catch=(Exception,),
    )
