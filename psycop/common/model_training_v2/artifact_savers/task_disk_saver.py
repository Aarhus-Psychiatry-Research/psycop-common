import pickle as pkl
from pathlib import Path

from psycop.common.model_training_v2.artifact_savers.base_artifact_saver import (
    BaselineArtifactSaver,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_trainer import (
    BaselineTrainer,
    TrainingResult,
)


@BaselineRegistry.artifact_savers.register("task_disk_saver")
class TaskDiskSaver(BaselineArtifactSaver):
    def __init__(self, experiment_path: str):
        self.run_path = Path(experiment_path)

    def save_artifact(
        self,
        trainer: BaselineTrainer,
        training_result: TrainingResult,  # noqa: ARG002
    ) -> None:
        self.run_path.mkdir(exist_ok=True, parents=True)
        self.save_path = self.run_path / "task.pkl"

        with self.save_path.open("wb") as f:
            pkl.dump(trainer.task, f)
