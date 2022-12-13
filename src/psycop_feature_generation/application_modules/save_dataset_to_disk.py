"""Utilities for saving a dataset to disk."""
import logging
from typing import Literal

import numpy as np
import pandas as pd

import psycop_feature_generation.loaders
from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.utils import write_df_to_file

log = logging.getLogger(__name__)


def save_split_to_disk(
    project_info: ProjectInfo,
    split_df: pd.DataFrame,
    split_name: str,
):
    """Save split to disk."""
    # Version table with current date and time
    filename = (
        f"{project_info.feature_set_prefix}_{split_name}.{project_info.dataset_format}"
    )
    log.info(f"Saving {filename} to disk")

    file_path = project_info.feature_set_path / filename

    write_df_to_file(df=split_df, file_path=file_path)

    log.info(f"{split_name}: Succesfully saved to {file_path}")


def filter_by_split_ids(
    flattened_df: pd.DataFrame,
    split_id_df: pd.DataFrame,
    split_name: str,
):
    """Filter dataframe by split ids."""
    # Find IDs which are in split_ids, but not in flattened_df
    flattened_df_ids = flattened_df["dw_ek_borger"].unique()
    split_ids: pd.Series = split_id_df["dw_ek_borger"].unique()

    ids_in_split_but_not_in_flattened_df = split_ids[
        ~np.isin(split_ids, flattened_df_ids)
    ]

    log.warning(
        f"{split_name}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df) / len(split_ids) * 100, 2)}%) ids which are in {split_name}_ids but not in flattened_df_ids, will get dropped during merge. If examining patients based on physical visits, see 'OBS: Patients without physical visits' on the wiki for more info.",
    )

    split_df = pd.merge(flattened_df, split_id_df, how="inner", validate="m:1")
    return split_df


def get_split_id_df(split_name: Literal["train", "val", "test"]) -> pd.DataFrame:
    """Get a dataframe with the splits ids."""
    split_id_df = psycop_feature_generation.loaders.raw.load_ids(
        split=split_name,
    )

    return split_id_df


@wandb_alert_on_exception
def split_and_save_dataset_to_disk(
    flattened_df: pd.DataFrame,
    project_info: ProjectInfo,
):
    """Split and save to disk.

    Args:
        flattened_df (pd.DataFrame): Flattened dataframe.
        project_info (ProjectInfo): Project info.
    """
    for split_name in ("train", "val", "test"):
        split_id_df = get_split_id_df(split_name=split_name)  # type: ignore

        split_df = filter_by_split_ids(
            flattened_df=flattened_df,
            split_name=split_name,
            split_id_df=split_id_df,
        )

        save_split_to_disk(
            project_info=project_info,
            split_df=split_df,
            split_name=split_name,
        )
