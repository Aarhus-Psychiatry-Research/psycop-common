"""Full configuration schema."""
from typing import Optional

from psycop_model_training.config_schemas.basemodel import BaseModel
from psycop_model_training.config_schemas.data import DataSchema
from psycop_model_training.config_schemas.debug import DebugConfSchema
from psycop_model_training.config_schemas.model import ModelConfSchema
from psycop_model_training.config_schemas.preprocessing import PreprocessingConfigSchema
from psycop_model_training.config_schemas.project import ProjectSchema
from psycop_model_training.config_schemas.train import TrainConfSchema


class FullConfigSchema(BaseModel):
    """A recipe for a full configuration object."""

    project: ProjectSchema
    data: DataSchema
    preprocessing: PreprocessingConfigSchema
    model: ModelConfSchema
    train: TrainConfSchema
    debug: Optional[DebugConfSchema]
