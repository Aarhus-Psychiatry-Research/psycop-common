from typing import Optional

import pandas as pd

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class SplitDataset(PSYCOPBaseModel):
    """A dataset split into train, test and optionally validation."""

    class Config:
        """Configuration for the dataclass to allow pd.DataFrame as type."""

        arbitrary_types_allowed = True

    train: pd.DataFrame
    test: Optional[pd.DataFrame] = None
    val: pd.DataFrame
