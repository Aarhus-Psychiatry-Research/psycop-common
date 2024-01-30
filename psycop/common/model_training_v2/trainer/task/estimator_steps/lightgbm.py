from typing import Literal

from lightgbm import LGBMClassifier

from ....config.baseline_registry import BaselineRegistry
from ..model_step import ModelStep


@BaselineRegistry.estimator_steps.register("lightgbm")
def lightgbm_classifier_step(
    num_leaves: int = 31, max_bin: int = 64, device_type: Literal["cpu", "gpu"] = "cpu"
) -> ModelStep:
    return (
        "lightgbm",
        LGBMClassifier(num_leaves=num_leaves, device_type=device_type, max_bin=max_bin),  # type: ignore
    )
