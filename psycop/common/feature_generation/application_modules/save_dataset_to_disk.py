"""Utilities for saving a dataset to disk."""
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop.common.feature_generation.loaders.raw.load_ids import SplitName, load_ids
from psycop.common.feature_generation.utils import write_df_to_file

log = logging.getLogger(__name__)


def save_chunk_to_disk(
    project_info: ProjectInfo,
    flattened_df_chunk: pd.DataFrame,
    chunk: int,
):
    """Save split to disk."""
    # Version table with current date and time
    filename = f"flattened_dataset_chunk_{chunk}.parquet"
    file_path = project_info.flattened_dataset_dir / filename  # type: ignore
    log.info(f"Saving {file_path} to disk")

    write_df_to_file(df=flattened_df_chunk, file_path=file_path)

    log.info(f"Chunk {chunk}: Succesfully saved to {file_path}")


def save_split_to_disk(
    split_df: pd.DataFrame,
    split_name: str,
    feature_set_dir: Path,
):
    """Save split to disk."""
    # Version table with current date and time
    filename = f"{split_name}.parquet"
    file_path = feature_set_dir / filename
    log.info(f"Saving {file_path} to disk")

    write_df_to_file(df=split_df, file_path=file_path)

    log.info(f"{split_name}: Succesfully saved to {file_path}")


def filter_by_split_ids(
    df_to_split: pd.DataFrame,
    split_id_df: pd.DataFrame,
    split_name: Union[list[str], str],
    split_id_col: str = "dw_ek_borger",
) -> pd.DataFrame:
    """Filter dataframe by split ids."""
    # Find IDs which are in split_ids, but not in flattened_df
    flattened_df_ids = df_to_split[split_id_col].unique()
    split_ids: pd.Series = split_id_df[split_id_col].unique()  # type: ignore

    ids_in_split_but_not_in_flattened_df = split_ids[
        ~np.isin(split_ids, flattened_df_ids)
    ]

    log.warning(
        f"{','.join(split_name)}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df) / len(split_ids) * 100, 2)}%) ids which are in {','.join(split_name)}_ids but not in flattened_df_ids, will get dropped during merge. If examining patients based on physical visits, see 'OBS: Patients without physical visits' on the wiki for more info.",
    )

    split_df = pd.merge(df_to_split, split_id_df, how="inner", validate="m:1")
    return split_df


def get_split_id_df(split_name: SplitName) -> pd.DataFrame:
    """Get a dataframe with the splits ids."""
    split_id_df = load_ids(
        split=split_name,
    )

    return split_id_df


@wandb_alert_on_exception
def split_and_save_dataset_to_disk(
    flattened_df: pd.DataFrame,
    project_info: ProjectInfo,
    feature_set_dir: Path,
    split_ids: Optional[dict[str, pd.DataFrame]] = None,
    split_names: Sequence[str] = ("train", "val", "test"),
):
    """Split and save to disk.

    Args:
        flattened_df (pd.DataFrame): Flattened dataframe.
        project_info (ProjectInfo): Project info.
        feature_set_dir (Path): Directory for saving feature sets
        split_ids (dict[str, pd.DataFrame]): Split ids.
        split_names (tuple[str], optional): Names of split to create. Defaults to ("train", "val", "test").
    """
    for split_name in split_names:
        match split_name:
            case "train":
                split_name = SplitName.TRAIN  # noqa: PLW2901
            case "val":
                split_name = SplitName.VALIDATION  # noqa: PLW2901
            case "test":
                split_name = SplitName.TEST  # noqa: PLW2901
            case _:
                raise ValueError(
                    f"Splitname {split_name} is not allowed, try from ['train', 'test', 'val']",
                )

        if not split_ids:
            split_id_df = get_split_id_df(split_name=split_name)
        else:
            split_id_df = split_ids[split_name.value]

        split_df = filter_by_split_ids(
            df_to_split=flattened_df,
            split_id_df=split_id_df,
            split_name=split_name.value,
            split_id_col=project_info.col_names.id,
        )

        save_split_to_disk(
            split_df=split_df,
            split_name=split_name.value,
            feature_set_dir=feature_set_dir,
        )
