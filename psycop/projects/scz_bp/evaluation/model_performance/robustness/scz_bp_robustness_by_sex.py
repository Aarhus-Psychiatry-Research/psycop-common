import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_plot import (
    scz_bp_plot_robustness,
)
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)


def scz_bp_auroc_by_sex(eval_ds: EvalDataset) -> pn.ggplot:
    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.is_female,  # type: ignore
        input_name="is_female",
        confidence_interval=True,
        bin_continuous_input=False,
    )

    bool_to_sex = {False: "Male", True: "Female"}
    df["Sex"] = df["is_female"].map(bool_to_sex).astype("category")

    return scz_bp_plot_robustness(df, x_column="Sex", line_y_col_name="auroc", xlab="Sex")


if __name__ == "__main__":
    best_experiment = "sczbp/text_only"
    best_pos_rate = 0.04

    best_eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(
        experiment_name=best_experiment, model_type="joint"
    )
    p = scz_bp_auroc_by_sex(best_eval_ds)
