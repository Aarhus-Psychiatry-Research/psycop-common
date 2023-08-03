"""Dataclasses for evaluation."""
from typing import Any, Optional, Union

import pandas as pd
import polars as pl
from numpy import float64

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def get_predictions_for_positive_rate(
    desired_positive_rate: float,
    y_hat_probs: pd.Series,
) -> tuple[pd.Series, Union[float, float64]]:
    positive_threshold = y_hat_probs.quantile(1 - desired_positive_rate)

    # Remap y_hat_probs to 0/1 based on positive rate threshold
    y_hat_int = pd.Series(
        (y_hat_probs >= positive_threshold).astype(int),
    )

    actual_positive_rate = y_hat_int.mean()

    return y_hat_int, actual_positive_rate  # type: ignore


def get_predictions_for_threshold(
    desired_threshold: float,
    y_hat_probs: pd.Series,
) -> tuple[pd.Series, float]:
    y_hat_int = (y_hat_probs > desired_threshold).astype(int)
    return y_hat_int, desired_threshold


class EvalDataset(PSYCOPBaseModel):
    """Evaluation dataset.

    Makes the interfaces of our evaluation functions simpler and
    consistent.
    """

    ids: pd.Series
    pred_time_uuids: pd.Series
    pred_timestamps: pd.Series
    outcome_timestamps: Optional[pd.Series] = None
    y: Union[pd.Series, pd.DataFrame]
    y_hat_probs: Union[pd.Series, pd.DataFrame]
    age: Optional[pd.Series] = None
    is_female: Optional[pd.Series] = None
    exclusion_timestamps: Optional[pd.Series] = None
    custom_columns: Optional[dict[str, pd.Series]] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.Config.allow_mutation = True

    def get_predictions_for_positive_rate(
        self,
        desired_positive_rate: float,
        y_hat_probs_column: Optional[str] = "y_hat_probs",
    ) -> tuple[pd.Series, Union[float, float64]]:
        """Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.


        Note that this won't always match the desired positive rate exactly for e.g tree-based models, where predicted probabilities are binned, but it'll get as close as possible.
        """
        if isinstance(self.y_hat_probs, pd.Series):
            self.y_hat_probs = self.y_hat_probs.to_frame(name="y_hat_probs")

        return get_predictions_for_positive_rate(
            desired_positive_rate=desired_positive_rate,
            y_hat_probs=self.y_hat_probs[y_hat_probs_column],
        )

    def get_predictions_for_threshold(
        self,
        desired_threshold: float,
        y_hat_probs_column: Optional[str] = "y_hat_probs",
    ) -> tuple[pd.Series, float]:
        """Turns predictions above `desired_threshold` to 1, rest to 0"""
        if isinstance(self.y_hat_probs, pd.Series):
            self.y_hat_probs = self.y_hat_probs.to_frame(name="y_hat_probs")

        return get_predictions_for_threshold(
            desired_threshold=desired_threshold,
            y_hat_probs=self.y_hat_probs[y_hat_probs_column],
        )

    def to_pandas(self) -> pd.DataFrame:
        """Converts to a dataframe. Ignores attributes that are not set and
        unpacks the columns in custom_columns.
        """
        as_dict = self.dict()
        # remove custom_column to avoid appending a row for each custom column
        as_dict.pop("custom_columns")
        df = pd.DataFrame(as_dict)
        if self.custom_columns is not None:
            for col_name, col in self.custom_columns.items():
                df[col_name] = col
        # remove columns which are not set
        df = df.dropna(axis=1, how="all")
        return df

    def to_polars(self) -> pl.DataFrame:
        return pl.from_pandas(self.to_pandas())


class PipeMetadata(PSYCOPBaseModel):
    """Metadata for a pipe.

    Currently only has feature_importances and selected_features, but
    makes it easy to add more.
    """

    feature_importances: Optional[dict[str, float]] = None
    selected_features: Optional[dict[str, int]] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.Config.allow_mutation = True


class ModelEvalData(PSYCOPBaseModel):
    """Dataclass for model evaluation data."""

    eval_dataset: EvalDataset
    cfg: FullConfigSchema
    pipe_metadata: Optional[PipeMetadata] = None
