"""Dataclasses used in the project."""
from typing import Optional

import pandas as pd
from omegaconf import DictConfig
from pydantic import BaseModel

# pylint: disable=missing-class-docstring, too-few-public-methods


class ModelEvalData(BaseModel):
    """Dataclass for model evaluation data."""

    class Config:
        arbitrary_types_allowed = True

    df: pd.DataFrame
    cfg: DictConfig
    feature_importance_dict: Optional[dict[str, float]] = None
