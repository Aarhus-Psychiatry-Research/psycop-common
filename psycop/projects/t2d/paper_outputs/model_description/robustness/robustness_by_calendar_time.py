import plotnine as pn

from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun


def t2d_auroc_by_quarter(run: T2DPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = create_roc_auc_by_absolute_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="Q",
        confidence_interval=True,
    )

    return t2d_plot_robustness(
        df,
        x_column="time_bin",
        line_y_col_name="auroc",
        xlab="Quarter",
    )


if __name__ == "__main__":
    t2d_auroc_by_quarter(run=get_best_eval_pipeline())
