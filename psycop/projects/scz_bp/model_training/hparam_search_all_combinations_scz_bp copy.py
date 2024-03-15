from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
# from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

if __name__ == "__main__":
    populate_baseline_registry()

    n_trials = 150
    n_jobs = 15
    for cfg_path in (Path(__file__).parent / "config").iterdir():    
        OptunaHyperParameterOptimization().from_file(
            cfg_path,
            study_name=cfg_path.name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            direction="maximize",
            catch=(Exception,),
        )
