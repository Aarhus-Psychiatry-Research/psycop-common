from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import PsycopMlflowRun
from psycop.projects.cvd.cohort_examination.table_one.model import table_one_model
from psycop.projects.cvd.cohort_examination.table_one.view import table_one_view


def table_one_facade(
    output_dir: Path, run: PsycopMlflowRun, sex_col_name: str = "pred_sex_female_fallback_0"
):
    model = table_one_model(run=run, sex_col_name=sex_col_name)
    view = table_one_view(model=model)
    view.to_csv(output_dir / "table_one.csv")
