import wandb

from psycop.common.model_training.utils.utils import create_wandb_folders
from psycop.common.sequence_models.loggers.base import Logger


class WandbLogger(Logger):
    def __init__(
        self,
        project_name: str,
        group: str,
        entity: str,
        run_name: str,
        config: dict[str, float | str],
    ) -> None:
        create_wandb_folders()

        wandb.init(
            project=f"{project_name}",
            reinit=True,
            mode="offline",
            group=group,
            config=config,
            entity=entity,
            name=run_name,
        )

    def log_metrics(self, metrics: dict[str, float]) -> None:
        wandb.log(metrics)
