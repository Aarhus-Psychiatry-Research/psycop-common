import plotnine as pn
from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun


def t2d_auroc_by_quarter(run: PipelineRun) -> pn.ggplot:
    print("Plotting AUROC by calendar time")
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = create_roc_auc_by_absolute_time_df(
        labels=eval_ds.y,
        y_hat_probs=eval_ds.y_hat_probs,
        timestamps=eval_ds.pred_timestamps,
        bin_period="Q",
        confidence_interval=True,
    )

    return t2d_plot_robustness(
        df,
        x_column="time_bin",
        line_y_col_name="auroc",
        xlab="Quarter",
        figure_file_name="t2d_auroc_by_calendar_time",
    )


if __name__ == "__main__":
    t2d_auroc_by_quarter(run=BEST_EVAL_PIPELINE)
