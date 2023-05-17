import pandas as pd
import plotnine as pn
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
    get_timedelta_df,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH, PN_THEME
from psycop.projects.t2d.utils.best_runs import ModelRun


def plot_sensitivity_by_time_to_event(df: pd.DataFrame) -> pn.ggplot:
    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="days_to_outcome_binned",
                y="sensitivity",
                ymin="sens_lower",
                ymax="sens_upper",
                color="actual_positive_rate",
            ),
        )
        + pn.geom_point()
        + pn.geom_line()
        + pn.geom_errorbar(width=0.2, size=0.5)
        + pn.labs(x="Months to outcome", y="n patients")
        + PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
        + pn.scale_color_brewer(type="qual", palette=2)
        + pn.labs(color="PPR")
        + pn.guides(color=pn.guide_legend(reverse=True))
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
        )
    )

    return p


def t2d_sensitivity_by_time_to_event(run: ModelRun) -> pn.ggplot:
    eval_ds = run.get_eval_dataset()

    dfs = []

    for ppr in [0.01, 0.03, 0.05]:
        df = get_sensitivity_by_timedelta_df(
            y=eval_ds.y,
            y_pred=eval_ds.get_predictions_for_positive_rate(desired_positive_rate=ppr)[
                0
            ],
            time_one=eval_ds.pred_timestamps,
            time_two=eval_ds.outcome_timestamps,
            direction="t2-t1",
            bins=range(0, 60, 6),
            bin_unit="M",
            bin_continuous_input=True,
            drop_na_events=True,
        )
        df["ppr"] = ppr
        dfs.append(df)

    plot_df = pd.concat(dfs)

    return plot_sensitivity_by_time_to_event(plot_df)


if __name__ == "__main__":
    p = t2d_sensitivity_by_time_to_event(run=EVAL_RUN)

    plot_path = FIGURES_PATH / "sensitivity_by_time_to_event.png"
    p.save(plot_path)
