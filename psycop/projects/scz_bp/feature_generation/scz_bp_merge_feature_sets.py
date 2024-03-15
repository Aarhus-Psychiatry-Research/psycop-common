"""Combine features sets (e.g. embedded text with structured feature sets)"""


from pathlib import Path

import pandas as pd

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training.config_schemas.data import DataSchema
from psycop.common.model_training.data_loader.data_loader import DataLoader


def load_all_splits(data_dir: Path) -> pd.DataFrame:
    return DataLoader(DataSchema(dir=data_dir), column_name_checker=None).load_dataset_from_dir(
        split_names=["train", "val", "test"]
    )


if __name__ == "__main__":
    structured_path = (
        OVARTACI_SHARED_DIR / "scz_bp" / "flattened_datasets" / "structured_predictors"
    )
    all_relevant_dfm_encoder_path = (
        OVARTACI_SHARED_DIR
        / "scz_bp"
        / "text_exp"
        / "flattened_datasets"
        / "text_exp_730_all_relevant_tfidf-1000"
    )

    combined = DataLoader(
        DataSchema(dir=structured_path), column_name_checker=None
    )._check_and_merge_feature_sets(  # type: ignore
        datasets=[load_all_splits(structured_path), load_all_splits(all_relevant_dfm_encoder_path)]
    )  # type: ignore
    # remove duplicated columns
    combined = combined.loc[:, ~combined.columns.duplicated()]  # type: ignore

    combined.to_parquet(
        path=OVARTACI_SHARED_DIR
        / "scz_bp"
        / "flattened_datasets"
        / "l1_l4-lookbehind_183_365_730-all_relevant_tfidf_1000_lookbehind_730.parquet"
    )
