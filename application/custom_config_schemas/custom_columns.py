from typing import Optional

import pandas as pd

from psycop_model_training.config_schemas.basemodel import BaseModel


class CustomColumns(BaseModel):
    """Custom columns to use in evaluation."""

    n_hba1c: Optional[pd.Series]
