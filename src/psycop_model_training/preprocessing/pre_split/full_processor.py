import pandas as pd

from psycop_model_training.preprocessing.pre_split.col_filterer import (
    PresSplitColFilterer,
)
from psycop_model_training.preprocessing.pre_split.col_transformer import (
    PresSplitColTransformer,
)
from psycop_model_training.preprocessing.pre_split.row_filterer import (
    PreSplitRowFilterer,
)
from psycop_model_training.utils.config_schemas import FullConfigSchema


class FullProcessor:
    """Uses all PresSplit preprocessors."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.row_filterer = PreSplitRowFilterer(cfg=cfg)
        self.col_filterer = PresSplitColFilterer(cfg=cfg)
        self.col_transformer = PresSplitColTransformer(cfg=cfg)

    def process_from_cfg(self, cfg: FullConfigSchema, df: pd.DataFrame):
        """Process a dataframe using the configuration."""
        df = self.row_filterer.filter_from_cfg(df=df)
        df = self.col_filterer.filter_from_cfg(df=df)
        df = self.col_transformer.transform_from_cfg(df=df)
        return df
