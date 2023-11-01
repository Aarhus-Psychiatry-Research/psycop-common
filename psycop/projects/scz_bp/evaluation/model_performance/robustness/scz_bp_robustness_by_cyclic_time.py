import plotnine as pn

from psycop.common.model_evaluation.binary.time.periodic_data import (
    roc_auc_by_periodic_time_df,
)
from psycop.projects.scz_bp.evaluation.pipeline_objects import PipelineRun
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)


def scz_bp_auroc_by_day_of_week(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="D",
    )

    return t2d_plot_robustness(
        df,
        x_column="time_bin",
        line_y_col_name="auroc",
        xlab="Day of Week",
    )


def scz_bp_auroc_by_month_of_year(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="M",
    )

    return t2d_plot_robustness(
        df,
        x_column="time_bin",
        line_y_col_name="auroc",
        xlab="Month of Year",
    )


if __name__ == "__main__":
    from psycop.projects.scz_bp.evaluation.model_selection.performance_by_group_lookahead_model_type import (
        DEVELOPMENT_PIPELINE_RUN,
    )

    scz_bp_auroc_by_day_of_week(run=DEVELOPMENT_PIPELINE_RUN)
    scz_bp_auroc_by_month_of_year(run=DEVELOPMENT_PIPELINE_RUN)
