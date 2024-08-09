from collections.abc import Sequence
from pathlib import Path

import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.projects.cvd.cohort_examination.table_one.model import table_one_model
from psycop.projects.cvd.cohort_examination.table_one.view import (
    ColumnOverride,
    RowCategory,
    cvd_table_one,
)


def table_one_facade(
    output_dir: Path, run: PsycopMlflowRun, sex_col_name: str = "pred_sex_female_fallback_0"
):
    model = table_one_model(run=run, sex_col_name=sex_col_name)
    view = cvd_table_one(model=model)
    view.to_excel(output_dir / "table_one.xlsx")
    view.to_csv(output_dir / "table_one.csv")


if __name__ == "__main__":
    table_one_facade(Path(), MlflowClientWrapper().get_run("CVD", "CVD 1, base, XGB"))
