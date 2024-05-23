"""Setup for the project."""

import logging
from pathlib import Path

from timeseriesflattener.v1.feature_specs.single_specs import BaseModel

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
