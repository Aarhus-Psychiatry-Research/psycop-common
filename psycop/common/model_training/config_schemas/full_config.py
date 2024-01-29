"""Full configuration schema."""
from typing import Optional

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel
from psycop.common.model_training.config_schemas.data import DataSchema
from psycop.common.model_training.config_schemas.debug import DebugConfSchema
from psycop.common.model_training.config_schemas.model import ModelConfSchema
from psycop.common.model_training.config_schemas.preprocessing import PreprocessingConfigSchema
from psycop.common.model_training.config_schemas.project import ProjectSchema
from psycop.common.model_training.config_schemas.train import TrainConfSchema


class FullConfigSchema(PSYCOPBaseModel):
    """A recipe for a full configuration object."""

    project: ProjectSchema
    data: DataSchema
    preprocessing: PreprocessingConfigSchema
    model: ModelConfSchema
    train: Optional[TrainConfSchema] = None

    n_crossval_splits: int = 5  # Number of splits to use for crossvalidation
    debug: Optional[DebugConfSchema] = None
