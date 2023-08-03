"""Preprocessing config schemas."""
from datetime import datetime
from typing import Literal, Optional, Union

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class FeatureSelectionSchema(PSYCOPBaseModel):
    """Configuration for feature selection methods."""

    name: Optional[str] = None
    # Which feature selection method to use.

    params: Optional[dict] = None
    # Parameters for the feature selection method.


class PreSplitPreprocessingConfigSchema(PSYCOPBaseModel):
    """Pre split preprocessing config."""

    drop_patient_if_exclusion_before_date: Optional[Union[str, datetime]] = None
    # Drop all visits from a patient if the outcome is before this date. If None, no patients are dropped.

    drop_visits_after_exclusion_timestamp: bool = True
    # Whether to drop visits for a given patietn after their exclusion timestamp. If False, no visits are dropped.

    drop_rows_after_outcome: bool = False
    # Whether to drop rows after the outcome has occurred. If False, no rows are dropped.

    convert_to_boolean: bool = False
    # Convert all prediction values (except gender) to boolean. Defaults to False. Useful as a sensitivity test, i.e. "is model performance based on whether blood samples are taken, or their values". If based purely on whether blood samples are taken, might indicate that it's just predicting whatever the doctor suspected.

    convert_booleans_to_int: bool = False
    # Whether to convert columns containing booleans to int

    negative_values_to_nan: bool = True
    # Whether to change negative values to NaN. Defaults to True since Chi2 cannot handle negative values. Can only be set to False if Chi2 is not used for feature selection.

    offset_so_no_negative_values: bool = False
    # Whether to offset values by the min value, so all values become non-negative. Defaults to False.

    drop_datetime_predictor_columns: bool = False
    # Whether to drop datetime columns prefixed with data.pred_prefix.
    # Typically, we don't want to use these as features, since they are unlikely to generalise into the future.

    convert_datetimes_to_ordinal: bool = False
    # Whether to convert datetimes to ordinal.

    # Looking ahead
    min_lookahead_days: int
    # Drop all prediction times where (max timestamp in the dataset) - (current timestamp) is less than min_lookahead_days

    min_age: Optional[float]  # Minimum age to include in the dataset

    min_prediction_time_date: Optional[Union[str, datetime]]
    # Drop all prediction times before this date.

    lookbehind_combination: list[int]
    # Which combination of features to use. Only uses features that have "within_X_days" in their column name, where X is any of the numbers in this list.

    keep_only_one_outcome_col: bool = True
    # Whether to keep only one outcome column, or all of them. If True, keeps the outcome column that matches the min_lookahead_days. If false, keeps all columns that match the prefix and min_lookahead_days.

    classification_objective: Literal["binary", "multilabel"] = "binary"
    # whether there are multiple outcome labels


class PostSplitPreprocessingConfigSchema(PSYCOPBaseModel):
    """Post split preprocessing config."""

    imputation_method: Optional[Literal["most_frequent", "mean", "median", "null"]]
    # How to replace missing values. Takes all values from the sklearn.impute.SimpleImputer class.
    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

    scaling: Optional[str]
    # Scaling applied to all predictors after imputation. Options include "z-score-normalization".

    feature_selection: FeatureSelectionSchema


class PreprocessingConfigSchema(PSYCOPBaseModel):
    """Preprocessing config."""

    pre_split: PreSplitPreprocessingConfigSchema
    post_split: PostSplitPreprocessingConfigSchema
