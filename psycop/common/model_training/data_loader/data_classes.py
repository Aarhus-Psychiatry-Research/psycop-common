from typing import Optional

import pandas as pd
from pydantic import ConfigDict

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class SplitDataset(PSYCOPBaseModel):
    """A dataset split into train, test and optionally validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: pd.DataFrame
    test: Optional[pd.DataFrame] = None
    val: pd.DataFrame
