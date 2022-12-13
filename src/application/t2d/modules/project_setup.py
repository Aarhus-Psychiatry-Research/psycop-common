import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence, Union

from timeseriesflattener.feature_spec_objects import BaseModel, PredictorSpec

from psycop_feature_generation.utils import RELATIVE_PROJECT_ROOT


class ProjectInfo(BaseModel):
    project_name: str
    project_path: Path
    feature_set_path: Path
    feature_set_id: str
    output_format: str: Literal["parquet", "csv"] = "parquet"

    def __init__():
        super().__init__()

        # Iterate over each attribute. If the attribute is a Path, create it if it does not exist.
        for attr in self.__dict__:
            if isinstance(attr, Path):
                attr.mkdir(exist_ok=True, parents=True)


def get_project_info(
    n_predictors: int,
    project_name: str,
) -> ProjectInfo:
    """Setup for main.

    Args:
        n_predictors (int): Number of predictors.
        feature_sets_path (Path): Path to feature sets.
        project_name (str): Name of project.
    Returns:
        tuple[Path, str]: Tuple of project path, and feature_set_id
    """
    proj_path = SHARED_RESOURCES_PATH / project_name

    current_user = Path().home().name
    feature_set_id = f"psycop_{project_name}_{current_user}_{n_predictors}_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    feature_set_path = create_feature_set_path(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    return ProjectInfo(
        project_path=proj_path,
        feature_set_path=feature_set_path,
        feature_set_id=feature_set_id,
        project_name=project_name,
    )


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


def init_wandb(
    wandb_project_name: str,
    predictor_specs: Sequence[PredictorSpec],
    feature_set_path: Union[Path, str],
) -> None:
    """Initialise wandb logging. Allows to use wandb to track progress, send
    Slack notifications if failing, and track logs.

    Args:
        wandb_project_name (str): Name of wandb project.
        predictor_specs (Iterable[dict[str, Any]]): List of predictor specs.
        feature_set_path (Union[Path, str]): Path to save dir.
    """

    feature_settings = {
        "feature_set_path": feature_set_path,
        "predictor_list": [spec.__dict__ for spec in predictor_specs],
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
            exist_ok=True, parents=True
        )

    wandb.init(project=wandb_project_name, config=feature_settings)
