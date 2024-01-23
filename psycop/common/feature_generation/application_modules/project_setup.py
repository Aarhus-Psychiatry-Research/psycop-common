"""Setup for the project."""
import logging
import sys
import tempfile
from pathlib import Path

import wandb
from timeseriesflattener.feature_specs.single_specs import BaseModel

from psycop.common.global_utils.paths import PSYCOP_PKG_ROOT

log = logging.getLogger(__name__)


class Prefixes(BaseModel):
    """Prefixes for feature specs."""

    predictor: str = "pred"
    outcome: str = "outc"
    eval: str = "eval"  # noqa


class ColNames(BaseModel):
    """Column names for feature specs."""

    timestamp: str = "timestamp"
    id: str = "dw_ek_borger"  # noqa


class ProjectInfo(BaseModel):
    """Collection of project info."""

    project_name: str
    project_path: Path
    prefix: Prefixes = Prefixes()
    col_names: ColNames = ColNames()

    @property
    def flattened_dataset_dir(self) -> Path:
        return self.project_path / "flattened_datasets"


def init_wandb(project_info: ProjectInfo) -> None:
    """Initialise wandb logging. Allows to use wandb to track progress, send
    Slack notifications if failing, and track logs.

    Args:
        project_info (ProjectInfo): Project info.
    """

    feature_settings = {"feature_set_path": project_info.flattened_dataset_dir}

    # on Overtaci, the wandb tmp directory is not automatically created,
    # so we create it here.
    # create debug-cli.one folders in /tmp and project dir
    if sys.platform == "win32":
        (Path(tempfile.gettempdir()) / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)
        (PSYCOP_PKG_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)

    wandb.init(
        project=f"{project_info.project_name}-feature-generation",
        config=feature_settings,
        mode="offline",
    )
