from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.projects.ect.cohort_examination.table_one.model import table_one_model
from psycop.projects.ect.cohort_examination.table_one.view import ect_table_one


def table_one_facade(
    output_dir: Path, cfg: PsycopConfig, sex_col_name: str = "pred_sex_female_fallback_0"
):
    model = table_one_model(cfg=cfg, sex_col_name=sex_col_name)
    view = ect_table_one(model=model)
    view.to_excel(output_dir / "table_one.xlsx")
    view.to_csv(output_dir / "table_one.csv")


if __name__ == "__main__":
    table_one_facade(
        Path("E:/shared_resources/ect/eval_runs/tables"),
        MlflowClientWrapper()
        .get_run(
            "ia_outpatient_all_features_training_best_run_evaluated_on_test", "auspicious-trout-251"
        )
        .get_config(),
    )
