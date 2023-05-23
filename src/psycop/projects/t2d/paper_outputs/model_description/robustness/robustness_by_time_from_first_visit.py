import plotnine as pn
import polars as pl
from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.common.model_evaluation.binary.time.periodic_data import (
    roc_auc_by_periodic_time_df,
)
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_auroc_by_timedelta_df,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)
from psycop.projects.t2d.utils.best_runs import ModelRun


def roc_auc_by_time_from_first_visit(run: ModelRun) -> pn.ggplot:
    print("Plotting AUC by time from first visit")
    eval_ds = run.get_eval_dataset()

    df = pl.DataFrame(
        {
            "id": eval_ds.ids,
            "y": eval_ds.y,
            "y_hat_probs": eval_ds.y_hat_probs,
            "pred_timestamp": eval_ds.pred_timestamps,
        }
    )

    first_visit = (
        df.sort("pred_timestamp", descending=False)
        .groupby("id")
        .head(1)
        .rename({"pred_timestamp": "first_visit_timestamp"})
    )

    df = df.join(
        first_visit.select(["first_visit_timestamp", "id"]), on="id"
    ).to_pandas()

    plot_df = get_auroc_by_timedelta_df(
        y=df["y"],
        y_pred_proba=df["y_hat_probs"],
        time_one=df["first_visit_timestamp"],
        time_two=df["pred_timestamp"],
        direction="t2-t1",
        bin_unit="M",
        bins=range(
            0,
            60,
            6,
        ),
    )

    return t2d_plot_robustness(
        plot_df,
        x_column="time_bin",
        line_y_col_name="auroc",
        xlab="Months since first visit",
        figure_file_name="t2d_auroc_by_time_from_first_visit",
    )


if __name__ == "__main__":
    roc_auc_by_time_from_first_visit(run=EVAL_RUN)
