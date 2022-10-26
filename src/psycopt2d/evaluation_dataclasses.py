from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

from psycopt2d.utils.configs import BaseModel


class EvalDataset(BaseModel):
    ids: pd.Series
    pred_timestamps: pd.Series
    outcome_timestamps: pd.Series
    y: pd.Series
    y_hat_probs: pd.Series
    y_hat_int: pd.Series
    age: pd.Series


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
    feature_importances: dict[str, float]
