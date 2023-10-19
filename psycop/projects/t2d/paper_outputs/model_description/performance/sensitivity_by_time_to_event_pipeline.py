import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.t2d.paper_outputs.config import (
    T2D_PN_THEME,
)
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun


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
        + pn.scale_x_discrete(reverse=True)
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.labs(x="Months to outcome", y="Sensitivity")
        + T2D_PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
        + pn.scale_color_brewer(type="qual", palette=2)
        + pn.labs(color="Predicted Positive Rate")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.3, 0.88),
        )
    )

    for value in df["actual_positive_rate"].unique():
        p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)

    return p


def t2d_plot_sensitivity_by_time_to_event(df: pd.DataFrame) -> pn.ggplot:
    categories = df["unit_from_event_binned"].dtype.categories[::-1]  # type: ignore
    df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
        new_categories=categories,
        ordered=True,  # type: ignore
    )

    p = _plot_sensitivity_by_time_to_event(df)

    return p


def sensitivity_by_time_to_event(eval_dataset: EvalDataset) -> pn.ggplot:
    dfs = []

    if eval_dataset.outcome_timestamps is None:
        raise ValueError(
            "The outcome timestamps must be provided in order to calculate the sensitivity by time to event.",
        )

    for ppr in [0.01, 0.03, 0.05]:
        df = get_sensitivity_by_timedelta_df(
            y=eval_dataset.y,
            y_hat=eval_dataset.get_predictions_for_positive_rate(
                desired_positive_rate=ppr,
            )[0],
            time_one=eval_dataset.pred_timestamps,
            time_two=eval_dataset.outcome_timestamps,
            direction="t2-t1",
            bins=range(0, 60, 6),
            bin_unit="M",
            bin_continuous_input=True,
            drop_na_events=True,
        )

        # Convert to string to allow distinct scales for color
        df["actual_positive_rate"] = str(ppr)
        dfs.append(df)

    plot_df = pd.concat(dfs)

    p = t2d_plot_sensitivity_by_time_to_event(plot_df)

    return p


def t2d_sensitivity_by_time_to_event(run: T2DPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    p = sensitivity_by_time_to_event(eval_dataset=eval_ds)

    p.save(run.paper_outputs.paths.figures, width=7, height=7)

    return p


if __name__ == "__main__":
    from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

    t2d_sensitivity_by_time_to_event(run=get_best_eval_pipeline())
