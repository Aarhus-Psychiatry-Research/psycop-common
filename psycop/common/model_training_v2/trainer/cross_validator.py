from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.base_trainer import (
    BaselineTrainer,
    TrainingResult,
)
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    PreprocessingPipeline,
)
from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.trainer.task.base_task import (
    BaselineTask,
)


class CrossValidatorTrainer(BaselineTrainer):
    def __init__(
        self,
        data: PolarsFrame,
        outcome_col_name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        task: BaselineTask,
        n_splits: int,
        logger: BaselineLogger,
    ):
        ...

    def train(self) -> TrainingResult:
        ...
