"""Dataclasses used in the project."""
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from psycopt2d.utils.omegaconf_to_pydantic_objects import FullConfig

# pylint: disable=missing-class-docstring, too-few-public-methods


class ModelEvalData(BaseModel):
    """Dataclass for model evaluation data."""

    class Config:
        arbitrary_types_allowed = True

    df: pd.DataFrame
    cfg: FullConfig
    feature_importance_dict: Optional[dict[str, float]] = None
