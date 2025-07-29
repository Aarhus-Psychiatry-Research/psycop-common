from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.projects.clozapine.model_eval.config import CLOZAPINE_EVAL_OUTPUT_DIR
from psycop.projects.clozapine.model_eval.table_one.model import table_one_model
from psycop.projects.clozapine.model_eval.table_one.view import clozapine_table_one


def table_one_facade(
    output_dir: Path, run: PsycopMlflowRun, sex_col_name: str = "pred_sex_female_fallback_0"
):
    output_dir = CLOZAPINE_EVAL_OUTPUT_DIR
    model = table_one_model(run=run, sex_col_name=sex_col_name)
    view = clozapine_table_one(model=model)
    view.to_excel(output_dir / "table_one_365d_lookahead_train_val.xlsx")
    view.to_csv(output_dir / "table_one_365d_lookahead_train_val.csv")


if __name__ == "__main__":
    table_one_facade(
        Path(__file__).parent,
        MlflowClientWrapper().get_run(
            "clozapine hparam, structured_text, xgboost, no lookbehind filter", "salty-shoat-573"
        ),
    )
