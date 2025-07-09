from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.restraint.evaluation.utils import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
    read_eval_df_from_disk,
)


def plotnine_sensitivity_by_tte(df: pd.DataFrame, title: str = "Temporal Sensitivity") -> pn.ggplot:
    categories = df["unit_from_event_binned"].dtype.categories[::-1]  # type: ignore
    df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
        new_categories=categories,
        ordered=True,  # type: ignore
    )

    df = df[df.unit_from_event_binned != "49+"]

    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="unit_from_event_binned",
                y="sensitivity",
                ymin="ci_lower",
                ymax="ci_upper",
                color="actual_positive_rate",
                shape="actual_positive_rate",
            ),
        )
        # + pn.scale_x_discrete(reverse=True)
        + pn.geom_point(size=2.5)
        + pn.geom_errorbar(width=0.05)
        + pn.labs(
            x="Hours to outcome",
            y="Sensitivity",
            title=title,
            color="Positive rate",
            shape="Positive rate",
        )
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(size=15),
            axis_text_y=pn.element_text(size=15),
            legend_position=(0.35, 0.9),
            legend_title_align="center",
            axis_ticks=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            # text=(pn.element_text(family="Times New Roman")),
            axis_title=pn.element_text(size=22),
            plot_title=pn.element_text(size=30, ha="center"),
            dpi=300,
        )
        + pn.scale_color_manual(values=["#669BBC", "#A8C686", "#F3A712"])
    )

    for value in df["actual_positive_rate"].unique():
        p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1, size=1)

    return p


def sensitivity_by_tte_model(
    df: pl.DataFrame, outcome_timestamps: pl.DataFrame, pprs: Sequence[float] = (0.01, 0.03, 0.05)
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

    dfs = []
    for ppr in pprs:
        df = get_sensitivity_by_timedelta_df(
            y=eval_dataset.y,  # type: ignore
            y_hat=get_predictions_for_positive_rate(
                desired_positive_rate=ppr, y_hat_probs=eval_dataset.y_hat_prob
            )[0],
            time_one=eval_dataset["timestamp"],
            time_two=eval_dataset["timestamp_outcome"],
            direction="t2-t1",
            bins=range(0, 49, 8),
            bin_unit="h",
            bin_continuous_input=True,
        )

        # Convert to string to allow distinct scales for color
        df["actual_positive_rate"] = str(ppr)
        dfs.append(df)

    plot_df = pd.concat(dfs)
    return plot_df


if __name__ == "__main__":
    save_dir = Path(__file__).parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_experiment = "restraint_text_hyper"
    eval_df = read_eval_df_from_disk(
        "E:/shared_resources/restraint/eval_runs/restraint_all_tuning_best_run_evaluated_on_test"
    )

    outcome_timestamps = pl.DataFrame(
        sql_load(
            "SELECT pred_times.dw_ek_borger, pred_time, first_mechanical_restraint as timestamp FROM fct.psycop_coercion_outcome_timestamps as pred_times LEFT JOIN fct.psycop_coercion_outcome_timestamps_2 as outc_times ON (pred_times.dw_ek_borger = outc_times.dw_ek_borger AND pred_times.datotid_start = outc_times.datotid_start)"
        ).drop_duplicates()
    )

    plotnine_sensitivity_by_tte(
        sensitivity_by_tte_model(df=eval_df, outcome_timestamps=outcome_timestamps)
    ).save(save_dir / "sensitivity_by_tte.png")
