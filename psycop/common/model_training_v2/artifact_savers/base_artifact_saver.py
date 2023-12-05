from typing import Protocol, runtime_checkable

from psycop.common.model_training_v2.trainer.base_trainer import (
    BaselineTrainer,
    TrainingResult,
)


@runtime_checkable
class BaselineArtifactSaver(Protocol):
    """Run at the end of training to save artifacts."""

    def save_artifact(
        self,
        trainer: BaselineTrainer,
        training_result: TrainingResult,
    ) -> None:
        ...
