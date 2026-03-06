from pathlib import Path
from typing import Literal, Mapping

import polars as pl

from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.t2d_extended.model_evaluation.table_one.model import prepare_table_one_dataset



def table_one_facade(
    output_dir: Path, datasets: Mapping[str, tuple[PsycopConfig, Literal["train", "val"]]], sex_col_name: str = "pred_sex_female_fallback_0",
):
    
    processed_dfs = []

    for dataset_name, (cfg, split) in datasets.items():

        df = prepare_table_one_dataset(
            cfg=cfg,
            sex_col_name=sex_col_name,
            dataset_name=dataset_name,
            split=split,
        )

        processed_dfs.append(df)


    all_processed = pl.concat(processed_dfs, how="vertical")


    fname = "count"

    all_processed.to_excel(output_dir / f"{fname}.xlsx")
    all_processed.to_csv(output_dir / f"{fname}.csv")

if __name__ == "__main__":

    exp_path = Path("E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation")

    dataset_dict = {
        "2013-2017": (PsycopConfig().from_disk(f"{exp_path}/2013-01-01_2018-01-01_2018-01-01_2018-12-31/config.cfg"), "train"),
        "2018": (PsycopConfig().from_disk(f"{exp_path}/2013-01-01_2018-01-01_2018-01-01_2018-12-31/config.cfg"), "val"),
        "2019": (PsycopConfig().from_disk(f"{exp_path}/2013-01-01_2018-01-01_2019-01-01_2019-12-31/config.cfg"), "val"),
        "2020": (PsycopConfig().from_disk(f"{exp_path}/2013-01-01_2018-01-01_2020-01-01_2020-12-31/config.cfg"), "val"),
        "2021": (PsycopConfig().from_disk(f"{exp_path}/2013-01-01_2018-01-01_2021-01-01_2021-12-31/config.cfg"), "val"),
        "2022": (PsycopConfig().from_disk(f"{exp_path}/2013-01-01_2018-01-01_2022-01-01_2022-12-31/config.cfg"), "val"),
        "2023": (PsycopConfig().from_disk(f"{exp_path}/2013-01-01_2018-01-01_2023-01-01_2023-12-31/config.cfg"), "val")
        }


    table_one_facade(
        output_dir=exp_path,
        datasets=dataset_dict,
    )


    print("hey")
