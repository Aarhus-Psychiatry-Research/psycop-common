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
        (Path(__file__).parent / "config" / "hparam_tuning" / "scz_bp_structured_only_logreg.cfg"),
        study_name="structured_only_log_reg_saga_scaler",
        n_trials=150,
        n_jobs=15,
        direction="maximize",
        catch=(Exception,),
    )
