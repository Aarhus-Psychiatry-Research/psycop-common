import plotnine as pn

from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.clozapine.model_eval.performance.robustness.clozapine_robustness_plot import (
    clozapine_plot_robustness,
)


def clozapine_auroc_by_quarter(eval_ds: EvalDataset) -> pn.ggplot:
    df = create_roc_auc_by_absolute_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="Q",
        confidence_interval=True,
    )

    return clozapine_plot_robustness(
        df, x_column="time_bin", line_y_col_name="auroc", xlab="Quarter"
    )
