import pandas as pd

from psycop_model_training.preprocessing.pre_split.processors.col_filter import (
    PresSplitColFilter,
)
from psycop_model_training.preprocessing.pre_split.processors.col_transformer import (
    PresSplitColTransformer,
)
from psycop_model_training.preprocessing.pre_split.processors.row_filter import (
    PreSplitRowFilter,
)


class FullProcessor:
    """Uses all PresSplit preprocessors."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.row_filterer = PreSplitRowFilter(cfg=cfg)
        self.col_filterer = PresSplitColFilter(cfg=cfg)
        self.col_transformer = PresSplitColTransformer(cfg=cfg)

    def process_from_cfg(self, dataset: pd.DataFrame):
        """Process a dataframe using the configuration."""
        dataset = self.row_filterer.filter(dataset=dataset)
        dataset = self.col_filterer.filter(dataset=dataset)
        dataset = self.col_transformer.transform_from_cfg(dataset=dataset)
        return dataset
