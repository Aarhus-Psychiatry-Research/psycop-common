from pydantic import BaseModel, DirectoryPath

from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer

from .loggers.base_logger import BaselineLogger


class BaselineSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    experiment_path: DirectoryPath  # Experiment_path must be a directory which exists
    logger: BaselineLogger
    trainer: BaselineTrainer


def train_baseline_model(cfg: BaselineSchema) -> float:
    cfg.logger.log_config(
        cfg.dict(),
    )  # Dict handling, might have to be flattened depending on the logger. Probably want all loggers to take flattened dicts.

    result = cfg.trainer.train()
    result.eval_dataset.to_disk(path=cfg.experiment_path)

    return result.metric.value
