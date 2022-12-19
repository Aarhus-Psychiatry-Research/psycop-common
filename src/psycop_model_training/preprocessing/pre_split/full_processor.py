"""Full processor for pre-split preprocessing."""
import pandas as pd

from psycop_model_training.preprocessing.pre_split.processors.col_filter import (
    PresSplitColFilter,
)
from psycop_model_training.preprocessing.pre_split.processors.row_filter import (
    PreSplitRowFilter,
)
from psycop_model_training.preprocessing.pre_split.processors.value_cleaner import (
    PreSplitValueCleaner,
)
from psycop_model_training.preprocessing.pre_split.processors.value_transformer import (
    PreSplitValueTransformer,
)


class FullProcessor:
    """Uses all PresSplit preprocessors."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.row_filterer = PreSplitRowFilter(cfg=cfg)
        self.col_filterer = PresSplitColFilter(cfg=cfg)
        self.value_transformer = PreSplitValueTransformer(cfg=cfg)
        self.value_cleaner = PreSplitValueCleaner(cfg=cfg)

    def process(self, dataset: pd.DataFrame):
        """Process a dataframe using the configuration."""
        dataset = self.value_cleaner.clean(dataset=dataset)
        dataset = self.value_transformer.transform(dataset=dataset)
        dataset = self.row_filterer.filter(dataset=dataset)
        dataset = self.col_filterer.filter(dataset=dataset)
        return dataset
