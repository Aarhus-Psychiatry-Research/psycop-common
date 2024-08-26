from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_positive_to_diagnosis_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.restraint.evaluation.evaluation_utils import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)


def plotnine_first_pos_pred_to_event(
    df: pd.DataFrame, title: str = "Detection-to-Event Interval"
) -> pn.ggplot:
    median_days = df["days_from_pred_to_event"].median()

    p = (
        pn.ggplot(
            df[df.days_from_pred_to_event <= 12], pn.aes(x="days_from_pred_to_event", fill="y")
        )  # type: ignore
        + pn.geom_density(alpha=0.8, fill="#B7C8B5")
        + pn.labs(x="Days until event", y="Proportion", title=title)
        + pn.scale_x_reverse(breaks=range(13))
        + pn.geom_segment(pn.aes(x=0, xend=0, y=0, yend=0.1525), linetype="solid", size=0.5)
        + pn.geom_text(
            pn.aes(x=median_days, y=0.125, label=median_days),
            format_string="Median days:\n{:.0f}",
            size=15,
            family="Times New Roman",
        )
        + pn.geom_segment(
            pn.aes(x=median_days, xend=median_days, y=0, yend=0.111), size=0.5, linetype="dashed"
        )
        + pn.theme_minimal()
        + pn.theme(
            axis_text_y=pn.element_blank(),
            axis_text_x=pn.element_text(size=15),
            axis_ticks=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            text=(pn.element_text(family="Times New Roman")),
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
    )

    return p


def first_pos_pred_to_model(
    df: pl.DataFrame, outcome_timestamps: pl.DataFrame, positive_rate: float = 0.05
) -> pd.DataFrame:
    eval_dataset = (
        parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(df))
        .with_columns(pl.col("timestamp").dt.cast_time_unit("ns"))
        .join(
            outcome_timestamps,
            left_on=["dw_ek_borger", "timestamp"],
            right_on=["dw_ek_borger", "pred_time"],
            suffix="_outcome",
            how="left",
        )
    ).to_pandas()

    df = pl.DataFrame(
        {
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=positive_rate, y_hat_probs=eval_dataset["y_hat_prob"]
            )[0],
            "y": eval_dataset["y"],
            "id": eval_dataset["dw_ek_borger"],
            "pred_timestamps": eval_dataset["timestamp"],
            "outcome_timestamps": eval_dataset["timestamp_outcome"],
        }
    )

    plot_df = get_time_from_first_positive_to_diagnosis_df(input_df=df.to_pandas())
    return plot_df


if __name__ == "__main__":
    save_dir = Path(__file__).parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    best_pos_rate = 0.05
    eval_df = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=best_experiment, metric="all_oof_BinaryAUROC")
        .eval_frame()
        .frame
    )
    outcome_timestamps = pl.DataFrame(
        sql_load(
            "SELECT pred_times.dw_ek_borger, pred_time, first_mechanical_restraint as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)"
        ).drop_duplicates()
    )

    plotnine_first_pos_pred_to_event(
        first_pos_pred_to_model(df=eval_df, outcome_timestamps=outcome_timestamps)
    ).save(save_dir / "first_pos_pred_to_event.png")
