from pathlib import Path

import pandas as pd
import plotnine as pn
from care_ml.model_evaluation.config import (
    COLOURS,
    EVAL_RUN,
    FIGURES_PATH,
    MODEL_NAME,
    PN_THEME,
    TEXT_EVAL_RUN,
    TEXT_FIGURES_PATH,
)
from care_ml.utils.best_runs import Run
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
)


def _plot_sensitivity_by_time_to_event(
    df: pd.DataFrame,
    path: Path,
    title: str,
) -> pn.ggplot:
    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="unit_from_event_binned",
                y="sensitivity",
                ymin="ci_lower",
                ymax="ci_upper",
            ),
        )
        + pn.geom_bar(
            pn.aes(y="proportion_in_bin"),
            stat="identity",
            position="identity",
            fill=COLOURS["blue"],
        )
        + pn.geom_text(
            pn.aes(y="proportion_in_bin", label="n_in_bin"),
            va="bottom",
            size=11,
        )
        + pn.scale_x_discrete(reverse=True)
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.labs(x="Hours to outcome", y="Sensitivity", title=title)
        + PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=90))
        + pn.labs(color="PPR")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.75, 0.2),
        )
        + pn.geom_path(group=1, size=0.5)
    )

    p.save(
        path / "sensitivity_by_time_to_event.png",
    )
    return p


def plot_sensitivity_by_time_to_event(
    df: pd.DataFrame,
    path: Path,
    title: str,
) -> pn.ggplot:
    categories = df["unit_from_event_binned"].dtype.categories[::-1]  # type: ignore
    df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
        new_categories=categories,
        ordered=True,  # type: ignore
    )

    df["proportion_in_bin"] = df["n_in_bin"] / sum(df.n_in_bin) * 7.5
    df["n_in_bin"] = df["n_in_bin"].astype(int)

    p = _plot_sensitivity_by_time_to_event(df, path, title)

    return p


def sensitivity_by_time_to_event(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    df = pd.DataFrame(
        {
            "y": eval_ds.y,
            "y_hat_probs": eval_ds.y_hat_probs,
            "pred_timestamps": eval_ds.pred_timestamps,
            "outcome_timestamps": eval_ds.outcome_timestamps,
        },
    )

    df = df[df.outcome_timestamps.notna()]

    df = get_sensitivity_by_timedelta_df(
        y=eval_ds.y,
        y_hat=eval_ds.get_predictions_for_positive_rate(
            desired_positive_rate=run.pos_rate,
        )[0],
        time_one=eval_ds.pred_timestamps,
        time_two=eval_ds.outcome_timestamps,
        direction="t2-t1",
        bins=range(0, 48, 4),
        bin_unit="h",
        bin_continuous_input=True,
        drop_na_events=True,
    )

    plot_sensitivity_by_time_to_event(
        df,
        path,
        f"{MODEL_NAME[run.name]} Performance by Time to Outcome",
    )


if __name__ == "__main__":
    sensitivity_by_time_to_event(EVAL_RUN, FIGURES_PATH)
    sensitivity_by_time_to_event(TEXT_EVAL_RUN, TEXT_FIGURES_PATH)
