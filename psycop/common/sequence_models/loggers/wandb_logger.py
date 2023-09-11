import wandb
from wandb.sdk.wandb_run import Run

from psycop.common.model_training.utils.utils import create_wandb_folders
from psycop.common.sequence_models.loggers.base import Logger


class WandbLogger(Logger):
    def __init__(
        self,
        project_name: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        entity: str | None = None,
        mode: str = "offline",
    ) -> None:
        create_wandb_folders()

        self.experiment: Run
        self.experiment = wandb.init(  # type: ignore
            project=f"{project_name}",
            reinit=True,
            mode=mode,
            group=group,
            entity=entity,
            name=run_name,
        )

        # get the run name from wandb
        self.run_name = self.experiment.name  # type: ignore

    def log_hyperparams(self, params: dict[str, float | str]) -> None:
        self.experiment.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        wandb.log(metrics)
