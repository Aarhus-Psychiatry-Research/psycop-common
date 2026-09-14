from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.projects.forced_admission_outpatient.table_one.model import table_one_model
from psycop.projects.forced_admission_outpatient.table_one.view import fao_table_one


def table_one_facade(output_dir: Path, run: PsycopMlflowRun, sex_col_name: str = "pred_sex_female"):
    output_dir = Path("E:/shared_resources/forced_admissions_outpatient/eval_runs")
    model = table_one_model(run=run, sex_col_name=sex_col_name)
    view = fao_table_one(model=model)

    out_dir = output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    view.to_excel(output_dir / "tables" / "table_1_fa_temp_val.xlsx")
    view.to_csv(output_dir / "tables" / "table_1_fa_temp_val.csv")


if __name__ == "__main__":
    table_one_facade(
        Path(__file__).parent,
        MlflowClientWrapper().get_run(
            "ia_outpatient_all_features_training_best_run_evaluated_on_test", "auspicious-trout-251"
        ),
    )
