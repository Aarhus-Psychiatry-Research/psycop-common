"""Dataclasses for evaluation."""
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from psycop_model_training.config_schemas.basemodel import BaseModel
from psycop_model_training.config_schemas.full_config import FullConfigSchema


class EvalDataset(BaseModel):
    """Evaluation dataset.

    Makes the interfaces of our evaluation functions simpler and
    consistent.
    """

    ids: pd.Series
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

    def get_predictions_for_positive_rate(self, positive_rate: float) -> pd.Series:
        """Converts predicted probabilities to discrete predictions (0/1) based no a desired positive rate."""
        positive_rate_threshold = self.y_hat_probs.quantile(positive_rate)

        # Remap y_hat_probs to 0/1 based on positive rate threshold
        y_hat_int = pd.Series(
            (self.y_hat_probs <= positive_rate_threshold).astype(int),
        )

        return y_hat_int


class ArtifactContainer(BaseModel):
    """A container for artifacts."""

    label: str
    # We're not a big fan of the naming here, super open to suggestions!
    # We need to keep the artifact and its labeled coupled, hence the
    # need for a container.
    artifact: Union[Path, pd.DataFrame]


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
