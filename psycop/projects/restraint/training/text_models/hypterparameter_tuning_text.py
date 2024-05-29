from pathlib import Path

import confection

from psycop.common.model_training_v2.config.baseline_pipeline import (
    train_baseline_model,
    train_baseline_model_from_cfg,
)
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
    cfg = confection.Config().from_disk(Path(__file__).parent / "restraint_hyperparam_text.cfg")
    train_baseline_model_from_cfg(cfg)  # modify key
    OptunaHyperParameterOptimization().from_cfg(
        cfg=cfg,
        study_name="test",
        n_trials=100,
        n_jobs=15,
        direction="maximize",
        catch=(Exception,),
        custom_populate_registry_fn=None,
    )
