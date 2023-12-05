import pickle as pkl
from pathlib import Path

from psycop.common.model_training_v2.artifact_savers.base_artifact_saver import (
    BaselineArtifactSaver,
)
from psycop.common.model_training_v2.trainer.base_trainer import (
    BaselineTrainer,
    TrainingResult,
)


class TaskDiskSaver(BaselineArtifactSaver):
    def __init__(self, experiment_path: str):
        self.experiment_path = Path(experiment_path)

    def save_artifact(
        self,
        trainer: BaselineTrainer,
        training_result: TrainingResult,  # noqa: ARG002
    ) -> None:
        self.experiment_path.mkdir(exist_ok=True, parents=True)
        self.save_path = self.experiment_path / "pipeline.pkl"

        with self.save_path.open("wb") as f:
            pkl.dump(trainer.task, f)
