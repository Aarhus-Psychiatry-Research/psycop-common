# imputation and scaling

# Exp 1: fit model on whole training data, decrease batch size a bit
# Exp 2: fit model on only positive cases
# Sample a large number of synthetic data points

# Save to parquet and concatenate with original data when training in main script


from pathlib import Path

import polars as pl
from confection import Config

from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


class DummyLogger(BaselineLogger):
    def info(self, message: str) -> None: ...

    def good(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...

    def fail(self, message: str) -> None: ...

    def log_metric(self, metric: CalculatedMetric) -> None: ...

    def log_config(self, config: Config) -> None: ...

    def log_artifact(self, local_path: Path) -> None: ...

    def log_dataset(self, dataframe: pl.DataFrame, filename: str) -> None: ...
