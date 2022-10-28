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


class WandbConf(BaseModel):
    """Configuration for weights and biases."""

    group: str
    mode: str
    entity: str


class WatcherConf(BaseModel):
    """Configuration for watchers."""

    archive_all: bool
    keep_alive_after_training_minutes: Union[int, float]
    n_runs_before_eval: int
    verbose: bool


class ProjectConf(BaseModel):
    """Project configuration."""

    wandb: WandbConf
    name: str = "psycopt2d"
    seed: int
    watcher: WatcherConf
    gpu: bool


class ColumnNames(BaseModel):
    """Column names in the data."""

    pred_prefix: str  # prefix of predictor columns
    pred_timestamp: str  # (str): Column name for prediction times
    outcome_timestamp: str  # (str): Column name for outcome timestamps
    id: str  # (str): Citizen colnames
    age: Optional[str]  # Name of the age column


class DataConf(BaseModel):
    """Data configuration."""

    n_training_samples: Optional[
        int
    ]  # (int, null): Number of training samples to use, defaults to null in which cases it uses all samples.
    dir: Union[Path, str]
    suffix: str  # File suffix to load.

    # Feature specs
    col_name: ColumnNames

    # Looking ahead
    min_lookahead_days: int  # (int): Drop all prediction times where (max timestamp in the dataset) - (current timestamp) is less than min_lookahead_days
    drop_patient_if_outcome_before_date: Optional[Union[str, datetime]]

    # Looking behind
    # (int): Drop all prediction times where (prediction_timestamp) - (min timestamp in the dataset) is less than min_lookbehind_days
    min_prediction_time_date: Optional[Union[str, datetime]]
    min_lookbehind_days: int
    max_lookbehind_days: Optional[int]
    lookbehind_combination: Optional[list[int]]


class FeatureSelectionConf(BaseModel):
    """Configuration for feature selection methods."""

    name: Optional[str]
    params: Optional[dict]


class PreprocessingConf(BaseModel):
    """Preprocessing config."""

    convert_to_boolean: bool  # (Boolean): Convert all prediction values (except gender) to boolean. Defaults to False
    convert_datetimes_to_ordinal: bool  # (str): Whether to convert datetimes to ordinal.
    imputation_method: Optional[str]  # (str): Options include "most_frequent"
    transform: Optional[
        str
    ]  # (str|null): Transformation applied to all predictors after imputation. Options include "z-score-normalization"
    feature_selection: FeatureSelectionConf


class ModelConf(BaseModel):
    """Model configuration."""

    name: str  # (str): Model, can currently take xgboost
    require_imputation: bool  # (bool): Whether the model requires imputation. (shouldn't this be false?)
    args: dict


class TrainConf(BaseModel):
    """Training configuration."""

    n_splits: int  # ? How do we handle whether to use crossvalidation or train/val splitting?
    n_trials_per_lookdirection_combination: int
    n_active_trainers: int  # Number of subprocesses to spawn when training
    gpu: bool


class EvalConf(BaseModel):
    """Evaluation config."""

    force: bool = False  # (bool): Whether to force evaluation even if wandb is not "run". Used for testing.

    threshold_percentiles: list[int]

    # top n features to plot. A table with all features is also logged
    top_n_feature_importances: int

    positive_rate_thresholds: list[int]
    save_model_predictions_on_overtaci: bool
    lookahead_bins: list[int]
    lookbehind_bins: list[int]


class FullConfigSchema(BaseModel):
    """A recipe for a full configuration object."""

    project: ProjectConf
    data: DataConf
    preprocessing: PreprocessingConf
    model: ModelConf
    train: TrainConf
    eval: EvalConf


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

        if not cfg.project.gpu and cfg.model.name == "xgboost":
            cfg.model.args["tree_method"] = "auto"

        return cfg


def load_cfg_as_pydantic(
    config_file_name,
    allow_mutation: bool = False,
) -> FullConfigSchema:
    """Load config as pydantic object."""
    cfg = load_cfg_as_omegaconf(config_file_name=config_file_name)

    return convert_omegaconf_to_pydantic_object(conf=cfg, allow_mutation=allow_mutation)
