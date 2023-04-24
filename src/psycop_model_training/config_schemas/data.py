from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union

from psycop_model_training.config_schemas.basemodel import BaseModel


class ColumnNamesSchema(BaseModel):
    """Column names in the data."""

    pred_timestamp: str = "timestamp"  # Column name for prediction times
    outcome_timestamp: str = "outc_timestamp"  # Column name for outcome timestamps
    id: str = "dw_ek_borger"  # Citizen colnames # noqa
    pred_time_uuid: str = "pred_time_uuid"  # Unique identifier for each prediction, useful if you later want to join the predictions with the original data.
    age: str = "pred_age_in_years"  # Name of the age column
    is_female: str = "pred_sex_female"  # Name of the sex column
    exclusion_timestamp: Optional[
        str
    ] = None  # Name of the exclusion timestamps column.
    # Drops all visits whose pred_timestamp <= exclusion_timestamp.
    custom_columns: Optional[list[str]] = None


class DataSchema(BaseModel):
    """Data configuration."""

    dir: Union[Path, str]  # Location of the dataset # noqa
    suffix: str = "parquet"  # File suffix to load.

    splits_for_training: Sequence[Literal["train", "val"]] = [
        "train",
        "val",
    ]  # splits to use for training
    splits_for_evaluation: Optional[Sequence[Literal["val", "test", None]]] = [
        None,
    ]  # splits to use for evaluation

    # Feature specs
    col_name: ColumnNamesSchema = ColumnNamesSchema()

    pred_prefix: str = "pred_"  # prefix of predictor columns
    outc_prefix: str = "outc_"  # prefix of outcome columns

    n_training_samples: Optional[int]
    # Number of training samples to use, defaults to null in which cases it uses all samples.
