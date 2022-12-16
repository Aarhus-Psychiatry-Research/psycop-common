from psycop_model_training.utils.config_schemas.conf_utils import BaseModel
from psycop_model_training.utils.config_schemas.data import DataSchema
from psycop_model_training.utils.config_schemas.eval import EvalConfSchema
from psycop_model_training.utils.config_schemas.model import ModelConfSchema
from psycop_model_training.utils.config_schemas.preprocessing import (
    PreprocessingConfigSchema,
)
from psycop_model_training.utils.config_schemas.project import ProjectSchema
from psycop_model_training.utils.config_schemas.train import TrainConfSchema


class FullConfigSchema(BaseModel):
    """A recipe for a full configuration object."""

    project: ProjectSchema
    data: DataSchema
    preprocessing: PreprocessingConfigSchema
    model: ModelConfSchema
    train: TrainConfSchema
    eval: EvalConfSchema
