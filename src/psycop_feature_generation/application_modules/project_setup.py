"""Setup for the project."""
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import wandb
from timeseriesflattener.feature_spec_objects import (  # pylint: disable=no-name-in-module
    BaseModel,
)

from psycop_feature_generation.utils import RELATIVE_PROJECT_ROOT, SHARED_RESOURCES_PATH

log = logging.getLogger(__name__)


class Prefixes(BaseModel):
    """Prefixes for feature specs."""

    predictor: str = "pred"
    outcome: str = "outc"
    eval: str = "eval"


class ColNames(BaseModel):
    """Column names for feature specs."""

    timestamp = "timestamp"
    id = "dw_ek_borger"


class ProjectInfo(BaseModel):
    """Collection of project info."""

    project_name: str
    project_path: Path
    feature_set_path: Path
    feature_set_prefix: str
    dataset_format: Literal["parquet", "csv"] = "parquet"
    prefix: Prefixes = Prefixes()
    col_names: ColNames = ColNames()

    def __init__(self, **data):
        super().__init__(**data)

        # Iterate over each attribute. If the attribute is a Path, create it if it does not exist.
        for attr in self.__dict__:
            if isinstance(attr, Path):
                attr.mkdir(exist_ok=True, parents=True)


def create_feature_set_path(
    proj_path: Path,
    feature_set_id: str,
) -> Path:
    """Create save directory.

    Args:
        proj_path (Path): Path to project.
        feature_set_id (str): Feature set id.

    Returns:
        Path: Path to sub directory.
    """

    # Split and save to disk
    # Create directory to store all files related to this run
    save_dir = proj_path / "feature_sets" / feature_set_id

    save_dir.mkdir(exist_ok=True, parents=True)

    return save_dir


def get_project_info(
    project_name: str,
) -> ProjectInfo:
    """Setup for main.

    Args:
        project_name (str): Name of project.
    Returns:
        tuple[Path, str]: Tuple of project path, and feature_set_id
    """
    log.info("Setting up project")
    proj_path = SHARED_RESOURCES_PATH / project_name

    current_user = Path().home().name
    feature_set_id = f"psycop_{project_name}_{current_user}_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    feature_set_path = create_feature_set_path(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    return ProjectInfo(
        project_path=proj_path,
        feature_set_path=feature_set_path,
        feature_set_prefix=feature_set_id,
        project_name=project_name,
    )


def init_wandb(
    project_info: ProjectInfo,
) -> None:
    """Initialise wandb logging. Allows to use wandb to track progress, send
    Slack notifications if failing, and track logs.

    Args:
        project_info (ProjectInfo): Project info.
    """

    feature_settings = {
        "feature_set_path": project_info.feature_set_path,
    }

    # on Overtaci, the wandb tmp directory is not automatically created,
    # so we create it here.
    # create debug-cli.one folders in /tmp and project dir
    if sys.platform == "win32":
        (Path(tempfile.gettempdir()) / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )
        (RELATIVE_PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )

    wandb.init(
        project=f"{project_info.project_name}-feature-generation",
        config=feature_settings,
    )
