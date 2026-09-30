import polars as pl

from psycop.common.feature_generation.loaders.raw.load_text import get_valid_text_sfi_names
from psycop.projects.forced_admission_inpatient_temp_val.feature_generation.modules.text_models.forced_adm_temp_val_text_model_paths import (
    TEXT_EMBEDDINGS_DIR,
)


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
    def load_embedded_text(filename: str) -> pl.DataFrame:
        """Loads embedded text (e.g. from sentence-transformers) from disk.

        Args:
            filename (str): Name of file to load from disk. Assumes file is
                located in TEXT_EMBEDDINGS_DIR.
        """

        embedded_text_df = pl.scan_parquet(TEXT_EMBEDDINGS_DIR / filename)

        return embedded_text_df.collect()
