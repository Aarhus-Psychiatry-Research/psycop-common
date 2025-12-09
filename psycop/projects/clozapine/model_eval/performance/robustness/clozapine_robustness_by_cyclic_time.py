import plotnine as pn

from psycop.common.model_evaluation.binary.time.periodic_data import roc_auc_by_periodic_time_df
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_plot import (
    clozapine_plot_robustness,
)


def clozapine_auroc_by_day_of_week(eval_ds: EvalDataset) -> pn.ggplot:
    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="D",
    )

    return clozapine_plot_robustness(
        df, x_column="time_bin", line_y_col_name="auroc", xlab="Day of Week"
    )


def clozapine_auroc_by_month_of_year(eval_ds: EvalDataset) -> pn.ggplot:
    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="M",
    )

    return clozapine_plot_robustness(
        df, x_column="time_bin", line_y_col_name="auroc", xlab="Month of Year"
    )
