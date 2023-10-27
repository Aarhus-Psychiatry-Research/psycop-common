import copy
import sys
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf
from random_word import RandomWords

from psycop.common.global_utils.config_utils import flatten_nested_dict
from psycop.common.global_utils.paths import PSYCOP_PKG_ROOT
from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.utils.utils import create_wandb_folders


class WandbHandler:
    """Class for handling wandb setup and logging."""

    def __init__(self, cfg: FullConfigSchema):
        self.cfg = cfg

        # Required on Windows because the wandb process is sometimes unable to initialise
        create_wandb_folders()

    def _unpack_pydantic_objects_in_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Takes a dict where some values are pydantic basemodels and
        recursively transforms them into a dict, ending up with only nested
        dicts and non-basemodel values."""
        for k, v in d.items():
            if isinstance(v, PSYCOPBaseModel):
                d[k] = v.__dict__
                d[k] = self._unpack_pydantic_objects_in_dict(d=d[k])
            elif isinstance(v, dict):
                continue

        return d

    def _get_cfg_as_dict(self) -> dict[str, Any]:
        """Get config as an unnested dict, with nesting represented as
        'key.val1.val2' in the key-name.

        Wandb does not allow for nested dicts in its configs.
        """
        if isinstance(self.cfg, DictConfig):
            cfg_as_dict = OmegaConf.to_container(self.cfg)
        else:
            cfg_copy = copy.deepcopy(self.cfg.__dict__)
            cfg_as_dict = self._unpack_pydantic_objects_in_dict(d=cfg_copy)

        flattened_dict = flatten_nested_dict(
            d=cfg_as_dict,  # type: ignore
            sep=".",
        )

        return flattened_dict

    def setup_wandb(self):
        """Setup wandb for the current run."""
        run_name = (
            None
            if self.cfg.project.wandb.mode != "offline"
            else RandomWords().get_random_word() + RandomWords().get_random_word()
        )

        # Create debug-cli on Windows, otherwise complains that it's missing
        if sys.platform == "win32":
            (PSYCOP_PKG_ROOT.parent.parent / "wandb" / "debug-cli.onerm").mkdir(
                exist_ok=True,
                parents=True,
            )

        wandb.init(
            project=f"{self.cfg.project.name}-baseline-model-training",
            reinit=True,
            mode=self.cfg.project.wandb.mode,
            group=self.cfg.project.wandb.group,
            config=self._get_cfg_as_dict(),
            entity=self.cfg.project.wandb.entity,
            name=run_name,
        )
