"""Preprocessing config schemas."""
from datetime import datetime
from typing import Literal, Optional, Union

from psycop_model_training.config_schemas.basemodel import BaseModel


class FeatureSelectionSchema(BaseModel):
    """Configuration for feature selection methods."""

    name: Optional[str] = None
    # Which feature selection method to use.

    params: Optional[dict] = None
    # Parameters for the feature selection method.


class PreSplitPreprocessingConfigSchema(BaseModel):
    """Pre split preprocessing config."""

    drop_patient_if_exclusion_before_date: Optional[Union[str, datetime]]
    # Drop all visits from a patient if the outcome is before this date. If None, no patients are dropped.

    drop_visits_after_exclusion_timestamp: Optional[bool]
    # Whether to drop visits for a given patietn after their exclusion timestamp. If False, no visits are dropped.

    convert_to_boolean: bool
    # Convert all prediction values (except gender) to boolean. Defaults to False. Useful as a sensitivty test, i.e. "is model performance based on whether blood samples are taken, or their values". If based purely on whether blood samples are taken, might indicate that it's just predicting whatever the doctor suspected.

    convert_booleans_to_int: bool
    # Whether to convert columns containing booleans to int

    drop_datetime_predictor_columns: bool
    # Whether to drop datetime columns prefixed with data.pred_prefix.
    # Typically, we don't want to use these as features, since they are unlikely to generalise into the future.

    convert_datetimes_to_ordinal: bool
    # Whether to convert datetimes to ordinal.

    min_age: Union[int, float]  # Minimum age to include in the dataset

    # Looking ahead
    min_lookahead_days: int
    # Drop all prediction times where (max timestamp in the dataset) - (current timestamp) is less than min_lookahead_days

    min_prediction_time_date: Optional[Union[str, datetime]]
    # Drop all prediction times before this date.

    lookbehind_combination: Optional[list[int]]
    # Which combination of features to use. Only uses features that have "within_X_days" in their column name, where X is any of the numbers in this list.


class PostSplitPreprocessingConfigSchema(BaseModel):
    """Post split preprocessing config."""

    imputation_method: Optional[Literal["most_frequent", "mean", "median", "null"]]
    # How to replace missing values. Takes all values from the sklearn.impute.SimpleImputer class.
    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

    scaling: Optional[str]
    # Scaling applied to all predictors after imputation. Options include "z-score-normalization".

    feature_selection: FeatureSelectionSchema


class PreprocessingConfigSchema(BaseModel):
    """Preprocessing config."""

    pre_split: PreSplitPreprocessingConfigSchema
    post_split: PostSplitPreprocessingConfigSchema
