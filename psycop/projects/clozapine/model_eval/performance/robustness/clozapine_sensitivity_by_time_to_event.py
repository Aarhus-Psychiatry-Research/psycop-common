from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
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
from psycop.projects.forced_admission_inpatient.model_eval.config import FA_PN_THEME


def _plot_sensitivity_by_time_to_event(df: pd.DataFrame) -> pn.ggplot:
    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="unit_from_event_binned",
                y="sensitivity",
                ymin="ci_lower",
                ymax="ci_upper",
                color="actual_positive_rate",
            ),
        )
        + pn.scale_x_discrete()
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.labs(x="Days to outcome", y="Sensitivity")
        + FA_PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
        + pn.scale_color_brewer(type="qual", palette=2)
        + pn.labs(color="Predicted Positive Rate")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_text=pn.element_text(size=10, color="black"),
            legend_position=(0.3, 0.88),
            axis_text=pn.element_text(size=10, weight="bold", color="black"),
            axis_title=pn.element_text(size=14, color="black"),
        )
    )

    for value in df["actual_positive_rate"].unique():
        p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)  # type: ignore
    return p


def clozapine_plot_sensitivity_by_time_to_event(df: pd.DataFrame) -> pn.ggplot:
    categories = df["unit_from_event_binned"].dtype.categories  # type: ignore
    df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
        new_categories=categories,
        ordered=True,  # type: ignore
    )

    p = _plot_sensitivity_by_time_to_event(df)

    return p


def sensitivity_by_time_to_event(
    eval_dataset: pl.DataFrame,
    outcome_timestamps: pl.DataFrame,
    positive_rates: Sequence[float] | None = None,
) -> pn.ggplot:  # type: ignore
    dfs = []

    if positive_rates is None:
        positive_rates = [0.01, 0.03, 0.05, 0.075, 0.1, 0.2]

    eval_dataset = eval_dataset.pipe(parse_dw_ek_borger_from_uuid).pipe(
        parse_timestamp_from_uuid, output_col_name="pred_time"
    )

    eval_dataset = eval_dataset.join(
        outcome_timestamps.rename({"timestamp": "outcome_timestamps"}).with_columns(),
        left_on=["dw_ek_borger"],
        right_on=["dw_ek_borger"],
        how="left",
        suffix="_outcome",
    )

    for ppr in positive_rates:
        y_hat_int, actual_positive_rate = get_predictions_for_positive_rate(
            desired_positive_rate=ppr,  # ✅ scalar
            y_hat_probs=eval_dataset["y_hat_prob"].to_pandas(),
        )

        df = get_sensitivity_by_timedelta_df(
            y=eval_dataset["y"],  # type: ignore
            y_hat=y_hat_int,
            time_one=eval_dataset["pred_time"],
            time_two=eval_dataset["outcome_timestamps"],
            direction="t2-t1",
            bins=range(0, 365, 60),
            bin_unit="D",
            bin_continuous_input=True,
            drop_na_events=True,
        )

        # Label by actual (or desired) positive rate
        df["actual_positive_rate"] = f"{actual_positive_rate:.3f}"
        dfs.append(df)

    plot_df = pd.concat(dfs)

    p = clozapine_plot_sensitivity_by_time_to_event(plot_df)

    return p


def clozapine_sensitivity_by_time_to_event(
    eval_df: pl.DataFrame,
    outcome_timestamps: pl.DataFrame,
    positive_rates: Sequence[float] | None = None,
) -> pn.ggplot:
    if positive_rates is None:
        positive_rates = [0.01, 0.03, 0.05, 0.075, 0.1, 0.2]

    p = sensitivity_by_time_to_event(
        eval_dataset=eval_df, outcome_timestamps=outcome_timestamps, positive_rates=positive_rates
    )

    p.save(save_dir / "clozapine_sensitivity_by_time_to_event.png", width=7, height=7, dpi=600)

    return p


if __name__ == "__main__":
    experiment_name = "clozapine hparam, structured_text_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split"
    best_pos_rate = 0.05
    eval_dir = (
        f"E:/shared_resources/clozapine/eval_runs/{experiment_name}_best_run_evaluated_on_test"
    )

    outcome_timestamps = pl.from_pandas(get_first_clozapine_prescription())

    eval_df = read_eval_df_from_disk(eval_dir)

    save_dir = Path("E:/shared_resources/clozapine/eval/figures")

    clozapine_sensitivity_by_time_to_event(eval_df=eval_df, outcome_timestamps=outcome_timestamps)
