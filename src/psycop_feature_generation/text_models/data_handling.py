from typing import Literal

import pandas as pd
from psycop_feature_generation.application_modules.save_dataset_to_disk import (
    filter_by_split_ids,
    get_split_id_df,
)
from psycop_feature_generation.loaders.raw.load_text import load_text_sfis


def load_text_split(
    text_sfi_names=str,
    split_name=Literal["train", "val"],
) -> pd.DataFrame:
    """Loads specified text sfi and only keeps data from the specified split"""

    text_df = load_text_sfis(text_sfi_names=text_sfi_names)

    split_id_df = get_split_id_df(split_name=split_name)

    text_split_df = filter_by_split_ids(
        df_to_split=text_df,
        split_id_df=split_id_df,
        split_name=split_name,
    )

    return text_split_df
