from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from psycop_model_training.utils.config_schemas import BaseModel


class CustomColNames(BaseModel):
    """All custom column names, i.e. columns that won't generalise across
    projects."""

    n_hba1c: str


class ColumnNamesSchema(BaseModel):
    """Column names in the data."""

    pred_timestamp: str  # Column name for prediction times
    outcome_timestamp: str  # Column name for outcome timestamps
    id: str  # Citizen colnames
    age: str  # Name of the age column
    exclusion_timestamp: str  # Name of the exclusion timestamps column.
    # Drops all visits whose pred_timestamp <= exclusion_timestamp.

    custom: Optional[CustomColNames] = None
    # Column names that are custom to the given prediction problem.


class DataSchema(BaseModel):
    """Data configuration."""

    n_training_samples: Optional[int]
    # Number of training samples to use, defaults to null in which cases it uses all samples.

    dir: Union[Path, str]  # Location of the dataset
    suffix: str  # File suffix to load.

    # Feature specs
    col_name: ColumnNamesSchema

    pred_prefix: str  # prefix of predictor columns
    outc_prefix: str  # prefix of outcome columns
