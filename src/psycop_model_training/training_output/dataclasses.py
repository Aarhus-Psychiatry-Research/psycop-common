"""Dataclasses for evaluation."""
from typing import Any, Optional, Union

import pandas as pd
from numpy import float64
from psycop_model_training.config_schemas.basemodel import BaseModel
from psycop_model_training.config_schemas.full_config import FullConfigSchema


class EvalDataset(BaseModel):
    """Evaluation dataset.

    Makes the interfaces of our evaluation functions simpler and
    consistent.
    """

    ids: pd.Series
    pred_time_uuids: pd.Series
    pred_timestamps: pd.Series
    outcome_timestamps: pd.Series
    y: pd.Series
    y_hat_probs: pd.Series
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
    ) -> tuple[pd.Series, Union[float, float64]]:
        """Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.


        Note that this won't always match the desired positive rate exactly for e.g tree-based models, where predicted probabilities are binned, but it'll get as close as possible.
        """
        positive_threshold = self.y_hat_probs.quantile(desired_positive_rate)

        # Remap y_hat_probs to 0/1 based on positive rate threshold
        y_hat_int = pd.Series(
            (self.y_hat_probs <= positive_threshold).astype(int),
        )

        actual_positive_rate = y_hat_int.mean()

        return y_hat_int, actual_positive_rate


class PipeMetadata(BaseModel):
    """Metadata for a pipe.

    Currently only has feature_importances and selected_features, but
    makes it easy to add more.
    """

    feature_importances: Optional[dict[str, float]] = None
    selected_features: Optional[dict[str, int]] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.Config.allow_mutation = True


class ModelEvalData(BaseModel):
    """Dataclass for model evaluation data."""

    eval_dataset: EvalDataset
    cfg: FullConfigSchema
    pipe_metadata: Optional[PipeMetadata] = None
