from pathlib import Path
from typing import Optional, Union

from psycop_model_training.config_schemas.basemodel import BaseModel


class ColumnNamesSchema(BaseModel):
    """Column names in the data."""

    pred_timestamp: str  # Column name for prediction times
    outcome_timestamp: str  # Column name for outcome timestamps
    id: str  # Citizen colnames # noqa
    age: str  # Name of the age column
    is_female: str  # Name of the sex column
    exclusion_timestamp: Optional[str]  # Name of the exclusion timestamps column.
    # Drops all visits whose pred_timestamp <= exclusion_timestamp.
    custom_columns: Optional[list[str]] = None


class DataSchema(BaseModel):
    """Data configuration."""

    n_training_samples: Optional[int]
    # Number of training samples to use, defaults to null in which cases it uses all samples.

    dir: Union[Path, str]  # Location of the dataset # noqa
    suffix: str  # File suffix to load.

    # Feature specs
    col_name: ColumnNamesSchema

    pred_prefix: str  # prefix of predictor columns
    outc_prefix: str  # prefix of outcome columns
