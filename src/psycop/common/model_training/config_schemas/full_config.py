"""Full configuration schema."""
from typing import Optional

from psycop.common.model_training.config_schemas.basemodel import BaseModel
from psycop.common.model_training.config_schemas.data import DataSchema
from psycop.common.model_training.config_schemas.debug import DebugConfSchema
from psycop.common.model_training.config_schemas.model import ModelConfSchema
from psycop.common.model_training.config_schemas.preprocessing import (
    PreprocessingConfigSchema,
)
from psycop.common.model_training.config_schemas.project import ProjectSchema
from psycop.common.model_training.config_schemas.train import TrainConfSchema


class FullConfigSchema(BaseModel):
    """A recipe for a full configuration object."""

    project: ProjectSchema
    data: DataSchema
    preprocessing: PreprocessingConfigSchema
    model: ModelConfSchema
    train: TrainConfSchema
    debug: Optional[DebugConfSchema]
