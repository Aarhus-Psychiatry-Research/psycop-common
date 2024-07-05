from sklearn.calibration import CalibratedClassifierCV

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


@BaselineRegistry.estimator_steps.register("estimator_calibrator")
def estimator_calibrator(model_step: ModelStep, n_calibration_folds: int = 5) -> ModelStep:
    return (
        f"calibrated_[{model_step[0]}]",
        CalibratedClassifierCV(estimator=model_step[1], cv=n_calibration_folds),
    )
