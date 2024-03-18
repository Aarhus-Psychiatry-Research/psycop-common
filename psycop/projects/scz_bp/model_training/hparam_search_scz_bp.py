from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)

if __name__ == "__main__":
    populate_baseline_registry()
    OptunaHyperParameterOptimization().from_file(
        (Path(__file__).parent / "config" / "scz_bp_structured_text.cfg"),
        study_name="sczbp-structured_text_",
        n_trials=300,
        n_jobs=15,
        direction="maximize",
        catch=(Exception,),
    )
