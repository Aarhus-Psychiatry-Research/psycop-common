from typing import Any, Dict

import wandb
from omegaconf import DictConfig, OmegaConf

from psycop_model_training.config_schemas.basemodel import BaseModel
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.config_schemas.project import WandbSchema
from psycop_model_training.utils.utils import create_wandb_folders, flatten_nested_dict


class WandbHandler:
    """Class for handling wandb setup and logging."""

    def __init__(self, cfg: FullConfigSchema):
        self.cfg = cfg

        # Required on Windows because the wandb process is sometimes unable to initialise
        create_wandb_folders()

    def _unpack_pydantic_objects_in_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Takes a dict where some values are pydantic basemodels and recursively transforms them into a dict, ending up with only nested dicts and non-basemodel values."""
        for k, v in d.items():
            if isinstance(v, BaseModel):
                d[k] = v.__dict__
                d[k] = self._unpack_pydantic_objects_in_dict(d=d[k])
            elif isinstance(v, dict):
                continue

        return d

    def _get_cfg_as_dict(self) -> dict[str, Any]:
        if isinstance(self.cfg, DictConfig):
            # Create flattened dict for logging to wandb
            # Wandb doesn't allow configs to be nested, so we
            # flatten it.
            cfg_as_dict = OmegaConf.to_container(self.cfg)

        else:
            # For testing, we can take a FullConfig object instead. Simplifies boilerplate.
            cfg_as_dict = self._unpack_pydantic_objects_in_dict(d=self.cfg.__dict__)

        flattened_dict = flatten_nested_dict(
            d=cfg_as_dict,
            sep=".",
        )  # type: ignore

        return flattened_dict

    def setup_wandb(self):
        """Setup wandb for the current run."""
        wandb.init(
            project=f"{self.cfg.project.name}-baseline-model-training",
            reinit=True,
            mode=self.cfg.project.wandb.mode,
            group=self.cfg.project.wandb.group,
            config=self._get_cfg_as_dict(),
            entity=self.cfg.project.wandb.entity,
        )
