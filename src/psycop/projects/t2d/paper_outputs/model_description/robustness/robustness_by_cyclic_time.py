import plotnine as pn
from psycop.common.model_evaluation.binary.time.periodic_data import (
    roc_auc_by_periodic_time_df,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun


def t2d_auroc_by_day_of_week(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,
        y_hat_probs=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="D",
    )

    return t2d_plot_robustness(
        df,
        x_column="time_bin",
        line_y_col_name="auroc",
        xlab="Day of Week",
        figure_file_name="t2d_auroc_by_day_of_week",
    )


def t2d_auroc_by_month_of_year(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,
        y_hat_probs=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="M",
    )

    return t2d_plot_robustness(
        df,
        x_column="time_bin",
        line_y_col_name="auroc",
        xlab="Month of Year",
        figure_file_name="t2d_auroc_by_month_of_year",
    )


if __name__ == "__main__":
    t2d_auroc_by_day_of_week(run=BEST_EVAL_PIPELINE)
    t2d_auroc_by_month_of_year(run=BEST_EVAL_PIPELINE)
