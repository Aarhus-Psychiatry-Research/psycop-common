from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

if __name__ == "__main__":
    populate_baseline_registry()

    outcome_dirs = ["bp_hparam", "scz_hparam"]

    n_trials = 150
    n_jobs = 25
    for outcome_dir in outcome_dirs:
        for cfg_path in (Path(__file__).parent / "config" / "individual_outcomes" / outcome_dir).iterdir():
            if "bp_structured_only" in str(cfg_path):
                print(f"Already processed {cfg_path}. Skipping")
                continue
            OptunaHyperParameterOptimization().from_file(
                cfg_path,
                study_name=cfg_path.stem,
                n_trials=n_trials,
                n_jobs=n_jobs,
                direction="maximize",
                catch=(Exception,),
                custom_populate_registry_fn=populate_scz_bp_registry,
            )   
