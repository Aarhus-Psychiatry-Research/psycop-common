from pydantic import BaseModel, DirectoryPath

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer


class ProjectInfo(PSYCOPBaseModel):
    experiment_path: DirectoryPath  # Experiment_path must be a directory which exists

class BaselineSchema(PSYCOPBaseModel):
    class Config:
        arbitrary_types_allowed = True

    project_info: ProjectInfo
    logger: BaselineLogger
    trainer: BaselineTrainer


        