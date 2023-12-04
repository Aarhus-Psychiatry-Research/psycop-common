from pathlib import Path

import pandas as pd
import polars as pl
from timeseriesflattener.feature_specs.single_specs import AnySpec

from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.application_modules.save_dataset_to_disk import (
    save_chunk_to_disk,
)


class ChunkedFeatureGenerator:
    @staticmethod
    def create_flattened_dataset_with_chunking(
        project_info: ProjectInfo,
        eligible_prediction_times: pd.DataFrame,
        feature_specs: list[AnySpec],
        chunksize: int = 400,
        quarantine_df: pd.DataFrame | None = None,
        quarantine_days: int | None = None,
    ) -> pd.DataFrame:
        """Generate features in chunks to avoid memory issues"""

        ChunkedFeatureGenerator.remove_files_from_dir(  # Ensures that all saved chunks from previous crashed run are removed
            project_info.flattened_dataset_dir,
        )

        print(f"Generating features in chunks of {chunksize}")
        for i in range(0, len(feature_specs), chunksize):
            print(f"Generating features for chunk {i} to {i+chunksize}")
            flattened_df_chunk = create_flattened_dataset(
                feature_specs=feature_specs[i : i + chunksize],
                prediction_times_df=eligible_prediction_times,
                drop_pred_times_with_insufficient_look_distance=False,
                project_info=project_info,
                quarantine_df=quarantine_df,
                quarantine_days=quarantine_days,
            )
            save_chunk_to_disk(project_info, flattened_df_chunk, i)

        print("Feature generation done. Merging feature sets...")
        df = ChunkedFeatureGenerator.merge_feature_sets_from_dirs(
            project_info.flattened_dataset_dir,
        )

        ChunkedFeatureGenerator.remove_files_from_dir(
            project_info.flattened_dataset_dir,
        )

        return df.to_pandas()

    @staticmethod
    def merge_feature_sets_from_dirs(
        feature_dir: Path,
    ) -> pl.DataFrame:
        """Merge all feature sets from source_dirs into target_dir"""
        dfs = ChunkedFeatureGenerator._read_chunk_dfs_from_dir(
            feature_dir=feature_dir,
        )
        df = ChunkedFeatureGenerator.merge_feature_dfs(dfs)

        return df

    @staticmethod
    def merge_feature_dfs(dfs: list[pl.DataFrame]) -> pl.DataFrame:
        shared_cols = ChunkedFeatureGenerator._find_shared_cols(dfs=dfs)
        # dataframes are sorted so can safely concat
        # need to remove duplicate columns
        dfs = ChunkedFeatureGenerator._remove_shared_cols_from_all_but_one_df(
            dfs=dfs,
            shared_cols=shared_cols,
        )
        df = pl.concat(dfs, how="horizontal")

        return df

    @staticmethod
    def _remove_shared_cols_from_all_but_one_df(
        dfs: list[pl.DataFrame],
        shared_cols: list[str],
    ) -> list[pl.DataFrame]:
        return [
            df.select(pl.exclude(shared_cols)) if i != 0 else df
            for i, df in enumerate(dfs)
        ]

    @staticmethod
    def _read_chunk_dfs_from_dir(
        feature_dir: Path,
    ) -> list[pl.DataFrame]:
        df_dirs = feature_dir.glob("flattened_dataset_chunk_*.parquet")

        return [pl.read_parquet(chunk_dir) for chunk_dir in df_dirs]  # type: ignore

    @staticmethod
    def _find_shared_cols(
        dfs: list[pl.DataFrame],
    ) -> list[str]:
        """Find columns that are present in all dataframes in dfs. Only checks the
        first two dataframes in dfs"""
        if len(dfs) < 2:
            return []

        cols = set(dfs[0].columns).intersection(dfs[1].columns)
        return list(cols)

    @staticmethod
    def remove_files_from_dir(
        feature_dir: Path,
    ):
        df_dirs = feature_dir.glob("flattened_dataset_chunk_*.parquet")

        for file in df_dirs:
            Path.unlink(Path(file))
