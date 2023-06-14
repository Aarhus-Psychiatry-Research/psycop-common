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
                color="actual_positive_rate",
            ),
        )
        + pn.scale_x_discrete(reverse=True)
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.labs(x="Hours to outcome", y="Sensitivity", title=title)
        + PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=90))
        + pn.scale_color_manual(
            {
                "0.01": COLOURS["blue"],
                "0.03": COLOURS["red"],
                "0.05": COLOURS["purple"],
            },
        )
        + pn.labs(color="PPR")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.75, 0.2),
        )
    )

    for value, colour in zip(
        df["actual_positive_rate"].unique(),
        [COLOURS["blue"], COLOURS["red"], COLOURS["purple"]],
    ):
        p += pn.geom_path(
            df[df["actual_positive_rate"] == value],
            group=1,
            colour=colour,
        )

    p.save(path / "sensitivity_by_ppr.png")
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

    p = _plot_sensitivity_by_time_to_event(df, path, title)

    return p


def sensitivity_by_time_to_event(run: Run, path: Path):
    eval_ds = run.get_eval_dataset()

    dfs = []

    for ppr in [0.01, 0.03, 0.05]:
        df = get_sensitivity_by_timedelta_df(
            y=eval_ds.y,
            y_hat=eval_ds.get_predictions_for_positive_rate(
                desired_positive_rate=ppr,
            )[0],
            time_one=eval_ds.pred_timestamps,
            time_two=eval_ds.outcome_timestamps,
            direction="t2-t1",
            bins=range(0, 48, 4),
            bin_unit="h",
            bin_continuous_input=True,
            drop_na_events=True,
        )

        # Convert to string to allow distinct scales for color
        df["actual_positive_rate"] = str(ppr)
        dfs.append(df)

    plot_df = pd.concat(dfs)

    plot_sensitivity_by_time_to_event(
        plot_df,
        path,
        f"Performance of Positive Rates for {MODEL_NAME[run.name]}",
    )


if __name__ == "__main__":
    sensitivity_by_time_to_event(EVAL_RUN, FIGURES_PATH)
    sensitivity_by_time_to_event(TEXT_EVAL_RUN, TEXT_FIGURES_PATH)
