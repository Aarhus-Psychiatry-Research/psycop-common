import pandas as pd

from psycop.common.feature_generation.loaders.raw.load_text import get_valid_text_sfi_names
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


class EmbeddedTextLoader:
    @staticmethod
    def _validate_sfi_names(text_sfi_names: list[str]) -> None:
        valid_sfi_names = get_valid_text_sfi_names()
        invalid_sfi_names = [
            sfi_name for sfi_name in text_sfi_names if sfi_name not in valid_sfi_names
        ]
        if invalid_sfi_names:
            raise ValueError(
                f"Invalid sfi names: {invalid_sfi_names}. Valid sfi names are: {valid_sfi_names}"
            )

    @staticmethod
    def load_embedded_text(
        embedding_view_name: str = "text_train_val_test_tfidf_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750",
        text_sfi_names: list[str] | None = None,
        n_rows: int | None = None,
        include_sfi_name: bool = False,
    ) -> pd.DataFrame:
        """Loads embedded text (e.g. from sentence-transformers) from disk.

        Args:
            embedding_view_name (str): Name of view to load from SQL database. Defaults to
                "text_train_val_test_tfidf_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750".
            text_sfi_names (list[str]): Which note types to load. See
                `get_all_valid_text_sfi_names()` for a list of valid note types. Defaults to None,
            n_rows (int | None): Number of rows to load. Defaults to None which
                loads all rows.
            include_sfi_name (bool): Whether to include column with sfi name ("overskrift"). Defaults to False.
        """

        if text_sfi_names:
            EmbeddedTextLoader._validate_sfi_names(text_sfi_names=text_sfi_names)

        # load embeddings
        view = f"[{embedding_view_name}]"
        sql = f"SELECT * FROM [fct].{view}"

        if text_sfi_names:
            sfis_to_keep = ", ".join("?" for _ in text_sfi_names)

            sql += f" WHERE overskrift IN ({sfis_to_keep})"

        if n_rows:
            sql += f" LIMIT {n_rows}"

        embeddings = sql_load(query=sql, server="BI-DPA-PROD", database="USR_PS_Forsk", n_rows=None)

        if include_sfi_name:
            embeddings = embeddings.drop(columns=["overskrift"])

        return embeddings
