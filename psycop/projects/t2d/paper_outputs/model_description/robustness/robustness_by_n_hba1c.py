import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun


def t2d_auroc_by_n_hba1c(
    run: T2DPipelineRun,
) -> pn.ggplot:
    """Plot performance by n hba1c"""
    eval_ds = run.pipeline_outputs.get_eval_dataset(
        custom_columns=["eval_hba1c_within_9999_days_count_fallback_nan"],
    )

    col_name = "eval_hba1c_within_9999_days_count_fallback_nan"
    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.custom_columns[col_name],  # type: ignore
        input_name=col_name,
        bins=[0, 2, 4, 6, 8, 10, 12],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    return t2d_plot_robustness(
        df,
        x_column="eval_hba1c_within_9999_days_count_fallback_nan_binned",
        line_y_col_name="auroc",
        xlab="n HbA1c measurements prior to visit",
    )


if __name__ == "__main__":
    t2d_auroc_by_n_hba1c(run=BEST_EVAL_PIPELINE)
