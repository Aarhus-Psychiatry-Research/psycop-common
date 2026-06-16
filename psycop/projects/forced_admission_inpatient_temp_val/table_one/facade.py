from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.projects.forced_admission_inpatient_temp_val.table_one.model import table_one_model
from psycop.projects.forced_admission_inpatient_temp_val.table_one.view import fa_temp_val_table_one


def table_one_facade(output_dir: Path, run: PsycopMlflowRun, sex_col_name: str = "pred_sex_female"):
    output_dir = Path("E:/shared_resources/forced_admissions_inpatient_temp_val/eval")
    model = table_one_model(run=run, sex_col_name=sex_col_name)
    view = fa_temp_val_table_one(model=model)

    out_dir = output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    view.to_excel(output_dir / "tables" / "table_1_fa_temp_val.xlsx")
    view.to_csv(output_dir / "tables" / "table_1_fa_temp_val.csv")


if __name__ == "__main__":
    table_one_facade(
        Path(__file__).parent,
        MlflowClientWrapper().get_run(
            "primary_eval_xgboost_structured_features_evaluated_on_full_temporal_split_updated_sex_column",
            "bright-cod-131",
        ),
    )
