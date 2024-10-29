from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.AcuteSomaticAdmission.model_training.populate_somatic_registry import (
    populate_with_somatic_registry,
)

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_somatic_registry()
    OptunaHyperParameterOptimization().from_file(
        (Path(__file__).parent / "somatic_hyperparam.cfg"),
        study_name="somatic_hyper_param_tot_experiments__",
        n_trials=100,
        n_jobs=30,
        direction="maximize",
        catch=(Exception,),
    )
