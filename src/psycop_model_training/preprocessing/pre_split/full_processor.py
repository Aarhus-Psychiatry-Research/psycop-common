"""Full processor for pre-split preprocessing."""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from psycop_model_training.config_schemas.data import DataSchema
from psycop_model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
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

log = logging.getLogger(__name__)

from joblib import Memory


class FullProcessor:
    """Uses all PresSplit preprocessors. Acts as an adapter in case we want to
    change the interfaces of its components.

    I.e. if we want to make PresSplitValueTransformer a class that takes
    a set of arguments instead of a FullConfig, we can do that without
    changing FullProcessor's interface.

    This means we can refactor without breaking the package for our
    users.
    """

    def __init__(
        self,
        pre_split_cfg: PreSplitPreprocessingConfigSchema,
        data_cfg: DataSchema,
    ):
        self.cfg = pre_split_cfg
        self.row_filterer = PreSplitRowFilter(
            pre_split_cfg=pre_split_cfg,
            data_cfg=data_cfg,
        )
        self.col_filterer = PresSplitColFilter(
            pre_split_cfg=pre_split_cfg,
            data_cfg=data_cfg,
        )
        self.value_transformer = PreSplitValueTransformer(
            pre_split_cfg=pre_split_cfg,
            data_cfg=data_cfg,
        )
        self.value_cleaner = PreSplitValueCleaner(
            pre_split_cfg=pre_split_cfg,
            data_cfg=data_cfg,
        )

    def process(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process a dataframe using the configuration."""
        dataset = self.value_cleaner.clean(dataset=dataset)
        dataset = self.value_transformer.transform(dataset=dataset)
        dataset = self.row_filterer.run_filter(dataset=dataset)
        dataset = self.col_filterer.run_filter(dataset=dataset)
        return dataset


def pre_split_process_full_dataset(
    dataset: pd.DataFrame,
    pre_split_cfg: PreSplitPreprocessingConfigSchema,
    data_cfg: DataSchema,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Process a full dataset using the configuration."""
    mem = Memory(location=cache_dir, verbose=1)

    @mem.cache(verbose=1)
    def __process_dataset(
        dataset: pd.DataFrame,
        pre_split_cfg: PreSplitPreprocessingConfigSchema,
        data_cfg: DataSchema,
    ) -> pd.DataFrame:
        processor = FullProcessor(
            pre_split_cfg=pre_split_cfg,
            data_cfg=data_cfg,
        )
        processed_dataset = processor.process(dataset=dataset)
        return processed_dataset

    processed_dataset = __process_dataset(dataset, pre_split_cfg, data_cfg)
    return processed_dataset
