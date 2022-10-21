"""Utilities for converting config yamls to pydantic objects.

Helpful because it makes them:
- Addressable with intellisense,
- Refactorable with IDEs,
- Easier to document with docstrings and
- Type checkable
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """Allow arbitrary types in all pydantic models."""

    class Config:
        """Allow arbitrary types."""

        arbitrary_types_allowed = True


class WatcherConf(BaseModel):
    """Configuration for watchers."""

    archive_all: bool
    keep_alive_after_training_minutes: Union[int, float]
    n_runs_before_eval: int


class ProjectConf(BaseModel):
    """Project configuration."""

    name: str = "psycopt2d"
    seed: int
    wandb_group: str
    wandb_mode: str
    wandb_entity: str
    watcher: WatcherConf
    gpu: bool


class DataConf(BaseModel):
    """Data configuration."""

    n_training_samples: Optional[
        int
    ]  # (int, null): Number of training samples to use, defaults to null in which cases it uses all samples.
    dir: Union[Path, str]
    suffix: str  # File suffix to load.

    # Feature specs
    pred_col_name_prefix: str  # (str): prefix of predictor columns
    pred_timestamp_col_name: str  # (str): Column name for prediction times
    outcome_timestamp_col_name: str  # (str): Column name for outcome timestamps
    id_col_name: str  # (str): Citizen colnames

    # Looking ahead
    lookahead_days: int  # (float): Number of days from prediction time to look ahead for the outcome.
    min_lookahead_days: Optional[
        int
    ]  # (int): Drop all prediction times where (max timestamp in the dataset) - (current timestamp) is less than min_lookahead_days
    min_lookbehind_days: Optional[int]
    drop_patient_if_outcome_before_date: Optional[Union[str, datetime]]

    # Looking behind
    # (int): Drop all prediction times where (prediction_timestamp) - (min timestamp in the dataset) is less than min_lookbehind_days
    min_prediction_time_date: Optional[Union[str, datetime]]
    lookbehind_combination: Optional[list[int]]


class PreprocessingConf(BaseModel):
    """Preprocessing config."""

    convert_to_boolean: bool  # (Boolean): Convert all prediction values (except gender) to boolean. Defaults to False
    convert_datetimes_to: bool  # (str): Options include ordinal or False
    imputation_method: Optional[str]  # (str): Options include "most_frequent"
    transform: Optional[
        str
    ]  # (str|null): Transformation applied to all predictors after imputation. Options include "z-score-normalization"


class ModelConf(BaseModel):
    """Model configuration."""

    model_name: str  # (str): Model, can currently take xgboost
    require_imputation: bool  # (bool): Whether the model requires imputation. (shouldn't this be false?)
    args: dict


class TrainConf(BaseModel):
    """Training configuration."""

    n_splits: int  # ? How do we handle whether to use crossvalidation or train/val splitting?
    n_trials_per_lookdirection_combination: int


class EvalConf(BaseModel):
    """Evaluation config."""

    threshold_percentiles: list[int]

    # top n features to plot. A table with all features is also logged
    top_n_feature_importances: int

    positive_rate_thresholds: list[int]
    save_model_predictions_on_overtaci: bool
    date_bins_ahead: list[int]
    date_bins_behind: list[int]


class FullConfig(BaseModel):
    """A full configuration object."""

    project: ProjectConf
    data: DataConf
    preprocessing: PreprocessingConf
    model: ModelConf
    train: TrainConf
    eval: EvalConf


def omegaconf_to_pydantic_objects(conf: DictConfig) -> FullConfig:
    """Converts an omegaconf DictConfig to a pydantic object.

    Args:
        conf (DictConfig): Omegaconf DictConfig

    Returns:
        FullConfig: Pydantic object
    """
    conf = OmegaConf.to_container(conf, resolve=True)  # type: ignore
    return FullConfig(**conf)
