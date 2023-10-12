from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class ColumnNamesSchema(PSYCOPBaseModel):
    """Column names in the data."""

    pred_timestamp: str = "timestamp"  # Column name for prediction times
    outcome_timestamp: Union[
        str,
        None,
    ] = "outc_timestamp"  # Column name for outcome timestamps
    id: str = "dw_ek_borger"  # Citizen colnames # noqa
    pred_time_uuid: str = "prediction_time_uuid"  # Unique identifier for each prediction, useful if you later want to join the predictions with the original data.
    age: Union[str, None] = "pred_age_in_years"  # Name of the age column
    is_female: Union[str, None] = "pred_sex_female"  # Name of the sex column
    exclusion_timestamp: Optional[
        str
    ] = None  # Name of the exclusion timestamps column.
    # Drops all visits whose pred_timestamp <= exclusion_timestamp.
    custom_columns: Optional[list[str]] = None


class DataSchema(PSYCOPBaseModel):
    """Data configuration."""

    dir: Union[Path, str]  # Location of the dataset # noqa
    suffix: str = "parquet"  # File suffix to load.

    splits_for_training: list[str] = [
        "train",
        "val",
    ]  # splits to use for training
    datasets_for_evaluation: Optional[Sequence[Union[Literal["val", "test"], str]]] = None
    # dataset to use for evaluation. Commonly, it would simply take either the val or test split 
    # corresponding to the train split, but a specific dataset may also be given when needed.
    # If None, crossvalidation is done on splits_for_training split(s).

    # Feature specs
    col_name: ColumnNamesSchema = ColumnNamesSchema()

    pred_prefix: str = "pred_"  # prefix of predictor columns
    outc_prefix: str = "outc_"  # prefix of outcome columns

    n_training_samples: Optional[int]
    # Number of training samples to use, defaults to null in which cases it uses all samples.
