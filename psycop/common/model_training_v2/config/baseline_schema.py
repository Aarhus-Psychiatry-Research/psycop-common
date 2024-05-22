from pydantic import BaseModel, ConfigDict, model_validator

from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer


class BaselineSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logger: BaselineLogger
    trainer: BaselineTrainer

    @model_validator(mode="after")  # type: ignore
    def update_loggers(self) -> "BaselineSchema":
        self.trainer.set_logger(self.logger)
        return self
