from dataclasses import dataclass

import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_time_from_first_positive_to_diagnosis_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)
from psycop.projects.t2d.paper_outputs.config import T2D_PN_THEME


def scz_bp_first_pred_to_event(eval_ds: EvalDataset, ppr: float) -> pn.ggplot:
    df = pd.DataFrame(
        {
            "pred": eval_ds.get_predictions_for_positive_rate(desired_positive_rate=ppr)[0],
            "y": eval_ds.y,
            "id": eval_ds.ids,
            "pred_timestamps": eval_ds.pred_timestamps,
            "outcome_timestamps": eval_ds.outcome_timestamps,
        }
    )

    plot_df = get_time_from_first_positive_to_diagnosis_df(input_df=df)

    median_years = plot_df["years_from_pred_to_event"].median()
    annotation_text = f"Median: {round(median_years, 1)!s} years"

    p = (
        pn.ggplot(plot_df, pn.aes(x="years_from_pred_to_event"))  # type: ignore
        # + pn.geom_histogram(binwidth=1, fill="orange") # noqa: ERA001
        + pn.geom_density()
        + pn.xlab("Years from first positive prediction\n to event")
        + pn.scale_x_reverse(breaks=range(int(plot_df["years_from_pred_to_event"].max() + 1)))
        + pn.ylab("n")
        + pn.geom_vline(xintercept=median_years, linetype="dashed", size=1)
        + pn.geom_text(
            pn.aes(x=median_years, y=40), label=annotation_text, ha="right", nudge_x=-0.3, size=11
        )
        + T2D_PN_THEME
    )

    return p


@dataclass
class PlotDfWithAnnotations:
    df: pd.DataFrame
    annotation_dict: dict[str, float]


def scz_bp_first_pred_to_event_stratified(
    eval_ds: EvalDataset, ppr: float
) -> PlotDfWithAnnotations:
    outcome2timestamp = {
        "SCZ": eval_ds.custom_columns["time_of_scz_diagnosis"],  # type: ignore
        "BP": eval_ds.custom_columns["time_of_bp_diagnosis"],  # type: ignore
        #   "both": eval_ds.outcome_timestamps, # noqa: ERA001# noqa: ERA001
    }

    dfs: list[pd.DataFrame] = []
    annotation_dict: dict[str, float] = {}
    for outcome, timestamps in outcome2timestamp.items():
        df = pd.DataFrame(
            {
                "pred": eval_ds.get_predictions_for_positive_rate(desired_positive_rate=ppr)[0],
                "y": eval_ds.y.copy(),
                "id": eval_ds.ids.copy(),
                "pred_timestamps": eval_ds.pred_timestamps.copy(),
                "outcome_timestamps": timestamps.copy(),  # type: ignore
                "outcome": outcome,
            }
        )
        plot_df = get_time_from_first_positive_to_diagnosis_df(input_df=df)
        median_years = plot_df["years_from_pred_to_event"].median()
        annotation_dict[outcome] = median_years
        plot_df["outcome_annotated"] = outcome + f" median: {median_years:.1f}"
        dfs.append(plot_df)

    plot_df = pd.concat(dfs)
    return PlotDfWithAnnotations(df=plot_df, annotation_dict=annotation_dict)


def plot_scz_bp_first_pred_to_event_stratified(eval_ds: EvalDataset, ppr: float) -> pn.ggplot:
    df_with_annotations = scz_bp_first_pred_to_event_stratified(eval_ds=eval_ds, ppr=ppr)
    plot_df = df_with_annotations.df
    annotation_dict = df_with_annotations.annotation_dict

    outcome2color: dict[str, str] = {"BP": "#669BBC", "SCZ": "#A8C686"}

    p = (
        pn.ggplot(plot_df, pn.aes(x="years_from_pred_to_event", fill="outcome_annotated"))  # type: ignore
        # + pn.geom_histogram(binwidth=1, alpha=0.7) # noqa: ERA001
        + pn.geom_density(alpha=0.8)
        + pn.xlab("Years from first positive prediction to event")
        + pn.scale_x_reverse(breaks=range(int(plot_df["years_from_pred_to_event"].max() + 1)))
        + pn.scale_fill_manual(values=list(outcome2color.values()))
        + pn.ylab("Density")
        + pn.theme_minimal()
        + pn.theme(
            legend_position=(0.35, 0.80),
            legend_direction="vertical",
            legend_title=pn.element_blank(),
            legend_text=pn.element_text(size=11),
            axis_title=pn.element_text(size=14),
            figure_size=(5, 5),
            axis_text=pn.element_text(size=9),
        )
    )

    for outcome, median in annotation_dict.items():
        p = p + pn.geom_vline(
            xintercept=median, linetype="dashed", size=1, color=outcome2color[outcome]
        )

    return p


if __name__ == "__main__":
    best_experiment = "sczbp/structured_text"
    best_pos_rate = 0.04
    eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=best_experiment)

    # p = scz_bp_first_pred_to_event(eval_ds=eval_ds, ppr=best_pos_rate) # noqa: ERA001
    p = scz_bp_first_pred_to_event_stratified(eval_ds=eval_ds, ppr=best_pos_rate)
