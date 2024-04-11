from imblearn.over_sampling import SMOTE

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


@BaselineRegistry.estimator_steps.register("smote")
def smote_step(sampling_strategy: float | str, n_jobs: int | None) -> ModelStep:
    return (
        "synthetic_data_augmentation",
        SMOTE(sampling_strategy=sampling_strategy, n_jobs=n_jobs),
    )
