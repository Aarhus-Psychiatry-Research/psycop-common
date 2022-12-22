from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema
from psycop_model_training.utils.config_schemas.project import WandbSchema
from psycop_model_training.utils.utils import create_wandb_folders, flatten_nested_dict


class WandbHandler:
    """Class for handling wandb setup and logging."""

    def __init__(self, cfg: FullConfigSchema, wandb_group: str):
        self.cfg = cfg
        self.wandb_group = wandb_group

        # Required on Windows because the wandb process is sometimes unable to initialise
        create_wandb_folders()

    def _get_cfg_as_dict(self) -> dict[str, Any]:
        if isinstance(self.cfg, DictConfig):
            # Create flattened dict for logging to wandb
            # Wandb doesn't allow configs to be nested, so we
            # flatten it.
            return flatten_nested_dict(
                OmegaConf.to_container(self.cfg),
                sep=".",
            )  # type: ignore
        else:
            # For testing, we can take a FullConfig object instead. Simplifies boilerplate.
            return self.cfg.__dict__

    def setup_wandb(self):
        """Setup wandb for the current run."""
        WandbSchema.init(
            project=f"{self.cfg.project.name}-baseline-model-training",
            reinit=True,
            mode=self.cfg.project.wandb.mode,
            group=self.wandb_group,
            config=self._get_cfg_as_dict(),
            entity=self.cfg.project.wandb.entity,
        )
