from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.training_method.preprocessing.pipeline import (
    PreprocessingPipeline,
)
from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.training_method.problem_type.problem_type_base import ProblemType
from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingMethod,
    TrainingResult,
)


class CrossValidatorTrainer(TrainingMethod):
    def __init__(
        self,
        data: PolarsFrame,
        outcome_col_name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        problem_type: ProblemType,
        n_splits: int,
        logger: BaselineLogger,
    ):
        ...

    def train(self) -> TrainingResult:
        ...
