import pandas as pd

from psycop_model_training.preprocessing.pre_split.processors.col_filter import (
    PresSplitColFilter,
)
from psycop_model_training.preprocessing.pre_split.processors.col_transformer import (
    PresSplitColTransformer,
)
from psycop_model_training.preprocessing.pre_split.processors.pre_split_pipeline import (
    apply_pre_split_pipeline,
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

    def process(self, dataset: pd.DataFrame):
        """Process a dataframe using the configuration."""
        dataset = self.col_transformer.transform(dataset=dataset)
        dataset = self.row_filterer.filter(dataset=dataset)
        dataset = self.col_filterer.filter(dataset=dataset)
        dataset = apply_pre_split_pipeline(cfg=self.cfg, data=dataset)
        return dataset
