import polars as pl

from psycop.common.feature_generation.loaders.raw.load_text import get_valid_text_sfi_names
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


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
    def _validate_input(text_sfi_names: list[str], filename: str):
        EmbeddedTextLoader._validate_sfi_names(text_sfi_names=text_sfi_names)
        if not (TEXT_EMBEDDINGS_DIR / filename).exists():
            raise FileNotFoundError(f"File {filename} not found in {TEXT_EMBEDDINGS_DIR}")

    @staticmethod
    def load_embedded_text(
        filename: str, text_sfi_names: list[str], include_sfi_name: bool, n_rows: int | None
    ) -> pl.DataFrame:
        """Loads embedded text (e.g. from sentence-transformers) from disk.

        Args:
            filename (str): Name of file to load from disk. Assumes file is
                located in TEXT_EMBEDDINGS_DIR.
            text_sfi_names (list[str]): Which note types to load. See
                `get_all_valid_text_sfi_names()` for a list of valid note types.
            include_sfi_name (bool): Whether to include column with sfi name
                ("overskrift").
            n_rows (int | None): Number of rows to load. Defaults to None which
                loads all rows."""
        EmbeddedTextLoader._validate_input(text_sfi_names=text_sfi_names, filename=filename)

        embedded_text_df = pl.scan_parquet(TEXT_EMBEDDINGS_DIR / filename).filter(
            pl.col("overskrift").is_in(text_sfi_names)
        )
        if n_rows is not None:
            embedded_text_df = embedded_text_df.head(n_rows)
        if not include_sfi_name:
            embedded_text_df = embedded_text_df.drop("overskrift")
        return embedded_text_df.collect()
