from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.projects.t2d_extended.paper_outputs.table_one.model import table_one_model_train, table_one_model_val
from psycop.projects.t2d_extended.paper_outputs.table_one.view import t2d_extended_table_one


def table_one_facade(
    output_dir: Path, run: PsycopMlflowRun, run_name: str, train_or_val: str, sex_col_name: str = "pred_sex_female_fallback_0",
):
    if train_or_val == "train":
        model = table_one_model_train(run=run, sex_col_name=sex_col_name)
    elif train_or_val == "val":
        model = table_one_model_val(run=run, sex_col_name=sex_col_name)
    view = t2d_extended_table_one(model=model)
    view.to_excel(output_dir / f"table_one_{run_name}_TRAIN.xlsx")
    view.to_csv(output_dir / f"table_one_{run_name}_TRAIN.csv")


if __name__ == "__main__":
    train_or_val = "train"
    run_name = "2018-01-01_2021-01-01_2021-12-31"
    table_one_facade(Path(), MlflowClientWrapper().get_run("T2D-extended, temporal validation", run_name), run_name=run_name, train_or_val=train_or_val)

    # TODO: make one for train set as well, make new table_one_model
