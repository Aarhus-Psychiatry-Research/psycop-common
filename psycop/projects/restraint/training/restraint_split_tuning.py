from pathlib import Path

from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.projects.restraint.training.populate_restraint_registry import (
    populate_with_restraint_registry,
)

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_restraint_registry()
    OptunaHyperParameterOptimization().from_file(
        (Path(__file__).parent / "configs" / "tuning" / "restraint_split_tuning_cont.cfg"),
        study_name="restraint_split_tuning_v2",
        n_trials=200,
        n_jobs=10,
        direction="maximize",
        catch=(),  # type: ignore
    )
