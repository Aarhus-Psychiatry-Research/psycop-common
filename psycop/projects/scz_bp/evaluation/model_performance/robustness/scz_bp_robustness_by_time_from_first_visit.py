import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_auroc_by_timedelta_df,
)
from psycop.projects.scz_bp.evaluation.pipeline_objects import PipelineRun
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)

#### TODO: re-run feature generation to get the first visit column and rewrite this


def scz_bp_auroc_by_time_from_first_visit(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = pl.DataFrame(
        {
            "id": eval_ds.ids,
            "y": eval_ds.y,
            "y_hat_probs": eval_ds.y_hat_probs,
            "pred_timestamp": eval_ds.pred_timestamps,
        },
    )

    first_visit = (
        df.sort("pred_timestamp", descending=False)
        .groupby("id")
        .head(1)
        .rename({"pred_timestamp": "first_visit_timestamp"})
    )

    df = df.join(
        first_visit.select(["first_visit_timestamp", "id"]),
        on="id",
    ).to_pandas()

    plot_df = get_auroc_by_timedelta_df(
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
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

    print("RERUN FEATURE GENERATION AND USE CORRECT TIMES FOR FIRST VISIT")

    return t2d_plot_robustness(
        plot_df,
        x_column="unit_from_event_binned",
        line_y_col_name="auroc",
        xlab="Months since first visit",
    )


if __name__ == "__main__":
    from psycop.projects.scz_bp.evaluation.model_selection.performance_by_group_lookahead_model_type import (
        DEVELOPMENT_PIPELINE_RUN,
    )

    scz_bp_auroc_by_time_from_first_visit(run=DEVELOPMENT_PIPELINE_RUN)
