from pathlib import Path

from pydantic import BaseModel

from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingMethod,
)

from .loggers.base_logger import BaselineLogger


class BaselineSchema(BaseModel):
    experiment_path: Path
    logger: BaselineLogger
    training_method: TrainingMethod


def train_baseline_model(cfg: BaselineSchema) -> float:
    cfg.logger.log_config(
        cfg.dict(),
    )  # Dict handling, might have to be flattened depending on the logger. Probably want all loggers to take flattened dicts.

    result = cfg.training_method.train()
    result.eval_dataset.to_disk(path=cfg.experiment_path)

    return result.metric
