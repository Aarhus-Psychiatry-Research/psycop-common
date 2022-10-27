from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd

from psycopt2d.utils.configs import BaseModel, FullConfig


class EvalDataset(BaseModel):
    class Config:
        allow_mutation = True

    ids: pd.Series
    pred_timestamps: pd.Series
    outcome_timestamps: pd.Series
    y: pd.Series
    y_hat_probs: pd.Series
    y_hat_int: pd.Series
    age: Optional[pd.Series]


class ArtifactSpecification(BaseModel):
    label: str
    artifact_generator_fn: Callable[[Any], Union[pd.DataFrame, Path]]
    kwargs: Optional[dict] = None


class ArtifactContainer(BaseModel):
    label: str
    # We're not a big fan of the naming here, super open to suggestions!
    # We need to keep the artifact and its labeled coupled, hence the
    # need for a container.
    artifact: Union[Path, pd.DataFrame]


class PipeMetadata(BaseModel):
    class Config:
        allow_mutation = True

    feature_importances: Optional[dict[str, float]] = None


class ModelEvalData(BaseModel):
    """Dataclass for model evaluation data."""

    eval_dataset: EvalDataset
    cfg: FullConfig
    pipe_metadata: Optional[PipeMetadata] = None
