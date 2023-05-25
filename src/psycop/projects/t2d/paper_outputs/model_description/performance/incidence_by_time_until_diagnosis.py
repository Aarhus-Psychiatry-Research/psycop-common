import pandas as pd
import plotnine as pn
import polars as pl
from psycop.projects.t2d.paper_outputs.config import (
    PN_THEME,
)
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun


def t2d_first_pred_to_event(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = pd.DataFrame(
        {
            "pred": eval_ds.get_predictions_for_positive_rate(
                desired_positive_rate=run.paper_outputs.pos_rate,
            )[0],
            "y": eval_ds.y,
            "id": eval_ds.ids,
            "pred_timestamp": eval_ds.pred_timestamps,
            "outcome_timestamps": eval_ds.outcome_timestamps,
        },
    )

    plot_df = get_time_from_first_positive_to_diagnosis_df(input_df=df)

    median_years = plot_df["years_from_pred_to_event"].median()
    annotation_text = f"Median: {str(round(median_years, 1))} years"

    p = (
        pn.ggplot(plot_df, pn.aes(x="years_from_pred_to_event"))  # type: ignore
        + pn.geom_histogram(binwidth=1, fill="orange")
        + pn.xlab("Years from first positive prediction\n to event")
        + pn.scale_x_reverse(
            breaks=range(0, int(plot_df["years_from_pred_to_event"].max() + 1)),
        )
        + pn.ylab("n")
        + pn.geom_vline(xintercept=median_years, linetype="dashed", size=1)
        + pn.geom_text(
            pn.aes(x=median_years, y=40),
            label=annotation_text,
            ha="right",
            nudge_x=-0.3,
            size=9,
        )
        + PN_THEME
    )

    p.save(run.paper_outputs.paths.figures / "first_pred_to_event.png")

    return p


def get_time_from_first_positive_to_diagnosis_df(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """input_df must contain columns:
        y: 1 or 0
        pred: 1 or 0
        id: Patient ID
        pred_timestamps: Timestamp of prediction
        outcome_timestamps: Timestamp of outcome

    Returns a dataframe with the additional columns:
        years_from_pred_to_event: Years from first positive prediction to event
        days_from_pred_to_event: Days from first positive prediction to event
    """
    df = pl.from_pandas(input_df).with_columns(
        (pl.col("outcome_timestamps") - pl.col("pred_timestamps")).alias(
            "time_from_pred_to_event"
        )
    )

    ever_positives = df.filter(
        pl.col("time_from_pred_to_event").is_not_null() & pl.col("pred") == 1
    )

    plot_df = (
        ever_positives.sort("time_from_pred_to_event", descending=True)
        .groupby("id")
        .head(1)
        .with_columns(
            (pl.col("time_from_pred_to_event").dt.days() / 365.25).alias(
                "years_from_pred_to_event",
            ),
            (pl.col("time_from_pred_to_event").dt.days()).alias(
                "days_from_pred_to_event",
            ),
        )
    ).to_pandas()

    return plot_df


if __name__ == "__main__":
    from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE

    t2d_first_pred_to_event(run=BEST_EVAL_PIPELINE).save(
        BEST_EVAL_PIPELINE.paper_outputs.paths.figures / "time_from_pred_to_event.png",
        width=5,
        height=5,
        dpi=600,
    )
