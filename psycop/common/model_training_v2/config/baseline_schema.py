from pydantic import BaseModel, ConfigDict, DirectoryPath, model_validator

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

    @model_validator(mode="after")
    def update_loggers(self) -> "BaselineSchema":
        self.trainer.set_logger(self.logger)
        return self
