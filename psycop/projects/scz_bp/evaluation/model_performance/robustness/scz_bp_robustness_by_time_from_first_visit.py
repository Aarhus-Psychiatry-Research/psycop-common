import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_auroc_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_plot import (
    scz_bp_plot_robustness,
)



def scz_bp_auroc_by_time_from_first_contact(eval_ds: EvalDataset) -> pn.ggplot:
    df = pl.DataFrame(
        {
            "id": eval_ds.ids,
            "y": eval_ds.y,
            "y_hat_probs": eval_ds.y_hat_probs,
            "pred_timestamp": eval_ds.pred_timestamps,
            "first_visit": eval_ds.custom_columns["first_visit"],  # type: ignore
        }
    )

    plot_df = get_auroc_by_timedelta_df(
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
        time_one=df["first_visit"],
        time_two=df["pred_timestamp"],
        direction="t2-t1",
        bin_unit="M",
        bins=range(0, 60, 6),
    )

    return scz_bp_plot_robustness(
        plot_df,
        x_column="unit_from_event_binned",
        line_y_col_name="auroc",
        xlab="Months since first visit",
    )
