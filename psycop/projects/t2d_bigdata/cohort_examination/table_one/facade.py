from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.projects.t2d_bigdata.cohort_examination.table_one.model import table_one_model
from psycop.projects.t2d_bigdata.cohort_examination.table_one.view import t2d_bigdata_table_one


def table_one_facade(
    output_dir: Path, run: PsycopMlflowRun, sex_col_name: str = "pred_sex_female_fallback_0"
):
    model = table_one_model(run=run, sex_col_name=sex_col_name)
    view = t2d_bigdata_table_one(model=model)
    view.to_excel(output_dir / "table_one.xlsx")
    view.to_csv(output_dir / "table_one.csv")


if __name__ == "__main__":
    table_one_facade(Path(), MlflowClientWrapper().get_run("T2D-bigdata", "kindly-squirrel-385"))
