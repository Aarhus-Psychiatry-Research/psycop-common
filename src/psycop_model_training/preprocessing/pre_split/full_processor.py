"""Full processor for pre-split preprocessing."""
import pandas as pd

from psycop_model_training.config_schemas.full_config import FullConfigSchema
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
    """Uses all PresSplit preprocessors. Acts as an adapter in case we want to
    change the interfaces of its components.

    I.e. if we want to make PresSplitValueTransformer a class that takes
    a set of arguments instead of a FullConfig, we can do that without
    changing FullProcessor's interface.

    This means we can refactor without breaking the package for our
    users.
    """

    def __init__(self, cfg: FullConfigSchema):
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
