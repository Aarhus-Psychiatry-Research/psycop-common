"""Pipeline for preprocessing all notes"""

from psycop_feature_generation.text_models.preprocessing import text_preprocessing
from psycop_feature_generation.loaders.raw.load_text import get_valid_text_sfi_names
from psycop_ml_utils.sql.writer import write_df_to_sql


def main():
    df = text_preprocessing(
        text_sfi_names=get_valid_text_sfi_names(),
        include_sfi_name=True,
        split_name=["train", "val"],
    )

    write_df_to_sql(
        df,
        table_name="psycop_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
        if_exists="replace",
        rows_per_chunk=5000,
    )


if __name__ == "__main__":
    main()
