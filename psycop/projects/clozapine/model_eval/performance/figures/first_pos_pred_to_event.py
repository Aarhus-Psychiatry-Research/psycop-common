from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_true_positive_to_outcome_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    get_first_clozapine_prescription,
)
from psycop.projects.clozapine.model_eval.utils import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
    read_eval_df_from_disk,
)


def plotnine_first_true_pos_pred_to_event(
    df: pd.DataFrame, title: str = "True-positive-to-Event Interval"
) -> pn.ggplot:
    median_days = df["days_from_true_positive_to_event"].median()

    p = (
        pn.ggplot(
            df[df.days_from_true_positive_to_event <= 365],
            pn.aes(x="days_from_true_positive_to_event", fill="y"),
        )  # type: ignore
        + pn.geom_density(alpha=0.8, fill="#B7C8B5")
        + pn.labs(x="Days until event", y="Proportion", title=title)
        + pn.scale_x_reverse()
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
    eval_dataset = df.pipe(parse_dw_ek_borger_from_uuid).pipe(
        parse_timestamp_from_uuid, output_col_name="pred_time"
    )

    eval_dataset = eval_dataset.join(
        outcome_timestamps.rename({"timestamp": "timestamp_outcome"}).with_columns(),
        left_on=["dw_ek_borger"],
        right_on=["dw_ek_borger"],
        how="left",
        suffix="_outcome",
    ).to_pandas()

    df = pl.DataFrame(
        {
            "pred": get_predictions_for_positive_rate(
                desired_positive_rate=positive_rate, y_hat_probs=eval_dataset["y_hat_prob"]
            )[0],
            "y": eval_dataset["y"],
            "id": eval_dataset["dw_ek_borger"],
            "pred_timestamps": eval_dataset["pred_time"],
            "outcome_timestamps": eval_dataset["timestamp_outcome"],
        }
    )

    plot_df = get_time_from_first_true_positive_to_outcome_df(input_df=df.to_pandas())
    return plot_df


if __name__ == "__main__":
    experiment_name = "clozapine hparam, structured_text_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split"
    best_pos_rate = 0.05
    eval_dir = (
        f"E:/shared_resources/clozapine/eval_runs/{experiment_name}_best_run_evaluated_on_test"
    )
    eval_df = read_eval_df_from_disk(eval_dir)

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")

    outcome_timestamps = pl.from_pandas(get_first_clozapine_prescription())

    plotnine_first_true_pos_pred_to_event(
        first_pos_pred_to_model(df=eval_df, outcome_timestamps=outcome_timestamps)
    ).save(save_dir / "clozapine_first_true_positive_to_event_xgboost.png")
