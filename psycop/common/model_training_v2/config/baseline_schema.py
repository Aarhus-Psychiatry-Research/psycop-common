from pydantic import BaseModel, ConfigDict, DirectoryPath

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer


class ProjectInfo(PSYCOPBaseModel):
    experiment_path: DirectoryPath  # Experiment_path must be a directory which exists


class BaselineSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_info: ProjectInfo
    logger: BaselineLogger
    trainer: BaselineTrainer

    def __post_init__(self):
        self.trainer.set_logger(self.logger)
