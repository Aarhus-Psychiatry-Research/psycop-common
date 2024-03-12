from pathlib import Path

from optuna import Trial

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry


@BaselineRegistry.suggesters.register("run_path_suggester")
class RunPathSuggester:
    def __init__(self, experiment_path: str):
        self.experiment_path = experiment_path

    def suggest_hyperparameters(self, trial: Trial) -> str:
        return str(Path(self.experiment_path) / f"run_{trial.number}")
