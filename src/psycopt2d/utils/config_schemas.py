"""Utilities for converting config yamls to pydantic objects. Helpful because
it makes them:

- Addressable with intellisense,
- Refactorable with IDEs,
- Easier to document with docstrings and
- Type checkable
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra


class BaseModel(PydanticBaseModel):
    """."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid

    def __init__(
        self,
        allow_mutation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Config.allow_mutation = allow_mutation


class WandbSchema(BaseModel):
    """Configuration for weights and biases."""

    group: str
    mode: str
    entity: str


class WatcherSchema(BaseModel):
    """Configuration for watchers."""

    archive_all: bool
    keep_alive_after_training_minutes: Union[int, float]
    n_runs_before_eval: int
    verbose: bool


class ProjectSchema(BaseModel):
    """Project configuration."""

    wandb: WandbSchema
    name: str = "psycopt2d"
    seed: int
    watcher: WatcherSchema
    gpu: bool


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

    min_age: Union[int, float]  # Minimum age to include in the dataset

    # Looking ahead
    min_lookahead_days: int
    # Drop all prediction times where (max timestamp in the dataset) - (current timestamp) is less than min_lookahead_days

    drop_patient_if_outcome_before_date: Optional[Union[str, datetime]]
    # Drop all visits from a patient if the outcome is before this date. If None, no patients are dropped.

    min_prediction_time_date: Optional[Union[str, datetime]]
    # Drop all prediction times before this date.

    min_lookbehind_days: int
    # Drop all prediction times where (prediction_timestamp) - (min timestamp in the dataset) is less than min_lookbehind_days

    lookbehind_combination: Optional[list[int]]
    # Which combination of features to use. Only uses features that have "within_X_days" in their column name, where X is any of the numbers in this list.


class FeatureSelectionSchema(BaseModel):
    """Configuration for feature selection methods."""

    name: Optional[str]
    # Which feature selection method to use.

    params: Optional[dict]
    # Parameters for the feature selection method.


class PreprocessingConfigSchema(BaseModel):
    """Preprocessing config."""

    convert_to_boolean: bool
    # Convert all prediction values (except gender) to boolean. Defaults to False. Useful as a sensitivty test, i.e. "is model performance based on whether blood samples are taken, or their values". If based purely on whether blood samples are taken, might indicate that it's just predicting whatever the doctor suspected.

    convert_datetimes_to_ordinal: bool
    # Whether to convert datetimes to ordinal.

    imputation_method: Optional[str]
    # How to replace missing values. Currently implemented are "most frequent".

    transform: Optional[str]
    # Transformation applied to all predictors after imputation. Options include "z-score-normalization"

    feature_selection: FeatureSelectionSchema


class ModelConfSchema(BaseModel):
    """Model configuration."""

    name: str  # Model, can currently take xgboost
    require_imputation: bool  # Whether the model requires imputation. (shouldn't this be false?)
    args: dict


class TrainConfSchema(BaseModel):
    """Training configuration."""

    n_splits: int  # ? How do we handle whether to use crossvalidation or train/val splitting?
    n_trials_per_lookdirection_combination: int
    n_active_trainers: int  # Number of subprocesses to spawn when training
    gpu: bool


class EvalConfSchema(BaseModel):
    """Evaluation config."""

    force: bool = False
    # Whether to force evaluation even if wandb is not "run". Used for testing.

    top_n_feature_importances: int
    # How many feature_importances to plot. Plots the most important n features. A table with all features is also logged.

    positive_rate_thresholds: list[int]
    # The threshold mapping a model's predicted probability to a binary outcome can be computed if we know, which positive rate we're targeting. We can't know beforehand which positive rate is best, beause it's a trade-off between false-positives and false-negatives. Therefore, we compute performacne for a range of positive rates.

    save_model_predictions_on_overtaci: bool

    lookahead_bins: list[int]
    # List of lookahead distances for plotting. Will create bins in between each distances. E.g. if specifying 1, 5, 10, will bin evaluation as follows: [0, 1], [1, 5], [5, 10], [10, inf].

    lookbehind_bins: list[int]
    # List of lookbehidn distances for plotting. Will create bins in between each distances. E.g. if specifying 1, 5, 10, will bin evaluation as follows: [0, 1], [1, 5], [5, 10], [10, inf].


class FullConfigSchema(BaseModel):
    """A recipe for a full configuration object."""

    project: ProjectSchema
    data: DataSchema
    preprocessing: PreprocessingConfigSchema
    model: ModelConfSchema
    train: TrainConfSchema
    eval: EvalConfSchema


def convert_omegaconf_to_pydantic_object(
    conf: DictConfig,
    allow_mutation: bool = False,
) -> FullConfigSchema:
    """Converts an omegaconf DictConfig to a pydantic object.

    Args:
        conf (DictConfig): Omegaconf DictConfig
        allow_mutation (bool, optional): Whether to make the pydantic object mutable. Defaults to False.
    Returns:
        FullConfig: Pydantic object
    """
    conf = OmegaConf.to_container(conf, resolve=True)  # type: ignore
    return FullConfigSchema(**conf, allow_mutation=allow_mutation)


def load_cfg_as_omegaconf(
    config_file_name: str,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Load config as omegaconf object."""
    with initialize(version_base=None, config_path="../config/"):
        if overrides:
            cfg = compose(
                config_name=config_file_name,
                overrides=overrides,
            )
        else:
            cfg = compose(
                config_name=config_file_name,
            )

        # Override the type so we can get autocomplete and renaming
        # correctly working
        cfg: FullConfigSchema = cfg  # type: ignore

        gpu = torch.cuda.is_available()
        if not gpu and cfg.model.name == "xgboost":
            cfg.model.args["tree_method"] = "auto"

        return cfg


def load_cfg_as_pydantic(
    config_file_name,
    allow_mutation: bool = False,
) -> FullConfigSchema:
    """Load config as pydantic object."""
    cfg = load_cfg_as_omegaconf(config_file_name=config_file_name)

    return convert_omegaconf_to_pydantic_object(conf=cfg, allow_mutation=allow_mutation)
