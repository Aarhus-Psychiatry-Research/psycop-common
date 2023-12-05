from typing import Protocol, runtime_checkable

from psycop.common.model_training_v2.trainer.base_trainer import (
    BaselineTrainer,
    TrainingResult,
)


@runtime_checkable
class BaselineArtifactSaver(Protocol):
    def save_artifact(
        self,
        trainer: BaselineTrainer,
        training_result: TrainingResult,
    ) -> None:
        ...
