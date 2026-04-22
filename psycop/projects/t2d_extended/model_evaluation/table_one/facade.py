"""Facade for generating table one for the T2D extended project.
Given the study design where each year is treated as a separate test set, patients with contacts in multiple years can appear in multiple data sets.
Consequently, the patient-level totals across data sets do not correspond to the total number of unique individuals in the study population.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import polars as pl

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.t2d_extended.model_evaluation.table_one.model import (
    TableOneModel,
    prepare_table_one_dataset,
)
from psycop.projects.t2d_extended.model_evaluation.table_one.view import t2d_extended_table_one


def table_one_facade(
    output_dir: Path,
    datasets: Mapping[str, tuple[PsycopConfig, Literal["train", "val"]]],
    sex_col_name: str = "pred_sex_female_fallback_0",
):
    all_processed = pl.concat(
        [
            prepare_table_one_dataset(
                cfg=cfg, sex_col_name=sex_col_name, dataset_name=dataset_name, split=split
            )
            for dataset_name, (cfg, split) in datasets.items()
        ],
        how="vertical",
    )

    model = TableOneModel(
        all_processed,
        allow_extra_columns=True,
        outcome_col_name=next(iter(datasets.values()))[0]["trainer"]["training_outcome_col_name"],
        sex_col_name=sex_col_name,
    )

    view = t2d_extended_table_one(model=model)
    view.to_csv(output_dir / "table_one.csv")


if __name__ == "__main__":
    exp_path = Path("E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation")

    dataset_dict = {
        "2013-2017": (
            PsycopConfig().from_disk(
                f"{exp_path}/2013-01-01_2018-01-01_2018-01-01_2018-12-31/config.cfg"
            ),
            "train",
        ),
        "2018": (
            PsycopConfig().from_disk(
                f"{exp_path}/2013-01-01_2018-01-01_2018-01-01_2018-12-31/config.cfg"
            ),
            "val",
        ),
        "2019": (
            PsycopConfig().from_disk(
                f"{exp_path}/2013-01-01_2018-01-01_2019-01-01_2019-12-31/config.cfg"
            ),
            "val",
        ),
        "2020": (
            PsycopConfig().from_disk(
                f"{exp_path}/2013-01-01_2018-01-01_2020-01-01_2020-12-31/config.cfg"
            ),
            "val",
        ),
        "2021": (
            PsycopConfig().from_disk(
                f"{exp_path}/2013-01-01_2018-01-01_2021-01-01_2021-12-31/config.cfg"
            ),
            "val",
        ),
        "2022": (
            PsycopConfig().from_disk(
                f"{exp_path}/2013-01-01_2018-01-01_2022-01-01_2022-12-31/config.cfg"
            ),
            "val",
        ),
        "2023": (
            PsycopConfig().from_disk(
                f"{exp_path}/2013-01-01_2018-01-01_2023-01-01_2023-12-31/config.cfg"
            ),
            "val",
        ),
    }

    table_one_facade(output_dir=exp_path, datasets=dataset_dict)
