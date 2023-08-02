from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import polars as pl
from timeseriesflattener.feature_specs.single_specs import AnySpec

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.scz_bp_config import (
    get_scz_bp_feature_specifications,
    get_scz_bp_project_info,
)


def generate_feature_set_in_chunks(
    project_info: ProjectInfo,
    eligible_prediction_times: pd.DataFrame,
    feature_specs: list[AnySpec],
    chunksize: int = 500,
) -> None:
    """Generate features in chunks to avoid memory issues with multiprocessing"""
    for i in range(0, len(feature_specs), chunksize):
        print(f"Generating features for chunk {i} to {i+chunksize}")
        init_wandb_and_generate_feature_set(
            project_info=project_info,
            eligible_prediction_times=eligible_prediction_times,
            feature_specs=feature_specs[i : i + chunksize],
        )
        move_contents_of_dir_to_dir(
            source_dir=project_info.project_path / "flattened_datasets",
            target_dir=project_info.project_path / f"flattened_datasets_chunk_{i}",
        )
    print("Feature generation done. Merging feature sets...")
    merge_feature_sets_from_dirs(
        source_dirs=project_info.project_path.glob("flattened_datasets_chunk_*"),
        target_dir=project_info.project_path / "flattened_datasets",
        splits=["train", "val", "test"],
    )


def move_contents_of_dir_to_dir(
    source_dir: Path,
    target_dir: Path,
) -> None:
    """Move all files and folders from source_dir to target_dir using pathlib"""
    target_dir.mkdir(exist_ok=True, parents=True)
    for file in source_dir.iterdir():
        file.rename(target_dir / file.name)


# TODO: refactor to smaller functions -- test whether the concatenations are correct
# against a join. Infer exclude cols 
def merge_feature_sets_from_dirs(
    source_dirs: Iterable[Path],
    target_dir: Path,
    splits: list[Literal["train", "val", "test"]] = ["train", "val", "test"],
) -> None:
    """Merge all feature sets from source_dirs into target_dir"""
    target_dir.mkdir(exist_ok=True, parents=True)
    for split in splits:
        dfs = [
            pl.read_parquet(source_dir / f"{split}.parquet")
            for source_dir in source_dirs
        ]
        # dataframes are sorted so can safely concat
        # need to remove duplicate columns
        dfs = [
            df.select(
                pl.exclude(
                    [
                        "dw_ek_borger",
                        "timestamp",
                        "prediction_time_uuid",
                        "date_of_birth",
                        "age",
                    ]
                )
            )
            for i, df in enumerate(dfs)
            if i != 0
        ]
        split_df = pl.concat(dfs, how="horizontal")
        split_df.write_parquet(target_dir / f"{split}.parquet")


if __name__ == "__main__":
    generate_feature_set_in_chunks(
        project_info=get_scz_bp_project_info(),
        eligible_prediction_times=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times.to_pandas(),
        feature_specs=get_scz_bp_feature_specifications()[:6],
        chunksize=2,
    )
