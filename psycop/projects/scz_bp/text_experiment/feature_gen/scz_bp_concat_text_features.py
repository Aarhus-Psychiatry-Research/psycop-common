from pathlib import Path

from psycop.common.model_training.config_schemas.data import DataSchema
from psycop.common.model_training.data_loader.data_loader import DataLoader

if __name__ == "__main__":
    text_exp_dir = Path("E:/shared_resources/scz_bp/text_exp/flattened_datasets")

    best_tfidf_schema = DataSchema(dir=text_exp_dir / "text_exp_730_aktuelt_psykisk_tfidf-1000")
    best_encoder_schema = DataSchema(
        dir=text_exp_dir / "text_exp_730_all_relevant_dfm-encoder-large"
    )

    tfidf_df = DataLoader(best_tfidf_schema, column_name_checker=None).load_dataset_from_dir(
        split_names=["train", "val", "test"]
    )
    encoder_df = DataLoader(best_encoder_schema, column_name_checker=None).load_dataset_from_dir(
        split_names=["train", "val", "test"]
    )

    combined = DataLoader(
        best_tfidf_schema, column_name_checker=None
    )._check_and_merge_feature_sets(datasets=[tfidf_df, encoder_df])  # type: ignore

    # remove duplicated columns
    combined = combined.loc[:, ~combined.columns.duplicated()]  # type: ignore

    combined.to_parquet(
        path=text_exp_dir
        / "text_exp_730_aktuelt_psykisk_tfidf_1000_and_all_relevant_dfm_encoder.parquet"
    )
