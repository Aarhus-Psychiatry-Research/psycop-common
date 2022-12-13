from typing import Literal

import numpy as np
import pandas as pd

import psycop_feature_generation.loaders
from application.t2d.modules.project_setup import ProjectInfo
from psycop_feature_generation.utils import write_df_to_file


def save_split_to_disk(
    project_info: ProjectInfo,
    split_df: pd.DataFrame,
    split_name: str,
):
    # Version table with current date and time
    filename = (
        f"{project_info.feature_set_id}_{split_name}.{project_info.dataset_format}"
    )
    msg.info(f"Saving {filename} to disk")

    file_path = project_info.feature_set_path / filename

    write_df_to_file(df=split_df, file_path=file_path)

    msg.good(f"{split_name}: Succesfully saved to {file_path}")


def filter_by_split_ids(
    flattened_df: pd.DataFrame,
    split_name: str,
    split_ids: pd.Series,
):
    """Filter dataframe by split ids."""
    msg = Printer(timestamp=True)

    # Find IDs which are in split_ids, but not in flattened_df
    flattened_df_ids = flattened_df["dw_ek_borger"].unique()

    ids_in_split_but_not_in_flattened_df = split_ids[
        ~np.isin(split_ids, flattened_df_ids)
    ]

    msg.warn(
        f"{split_name}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df) / len(split_ids) * 100, 2)}%) ids which are in {split_name}_ids but not in flattened_df_ids, will get dropped during merge. If examining patients based on physical visits, see 'OBS: Patients without physical visits' on the wiki for more info.",
    )

    split_df = pd.merge(flattened_df, df_split_ids, how="inner", validate="m:1")
    return split_df


def get_split_ids(split_name: Literal["train", "val", "test"]):
    df_split_ids = psycop_feature_generation.loaders.raw.load_ids(
        split=split_name,
    )

    split_ids: pd.Series = df_split_ids["dw_ek_borger"].unique()
    return split_ids


def split_and_save_dataset_to_disk(
    flattened_df: pd.DataFrame,
    project_info: ProjectInfo,
):
    """Split and save to disk.

    Args:
        flattened_df (pd.DataFrame): Flattened dataframe.
        project_info (ProjectInfo): Project info.
    """
    for split_name in ["train", "val", "test"]:
        split_ids = get_split_ids(split_name=split_name)

        split_df = filter_by_split_ids(
            flattened_df=flattened_df,
            split_name=split_name,
            split_ids=split_ids,
        )

        save_split_to_disk(
            project_info=project_info,
            split_df=split_df,
            split_name=split_name,
        )
