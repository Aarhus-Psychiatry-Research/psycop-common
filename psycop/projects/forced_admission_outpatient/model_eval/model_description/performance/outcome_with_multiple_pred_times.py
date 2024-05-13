import pandas as pd
import patchworklib as pw
import plotnine as pn
from wasabi import Printer

from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.forced_admission_outpatient.model_eval.config import BEST_POS_RATE, FA_PN_THEME
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)

msg = Printer(timestamp=True)


def _get_prediction_times_with_outcome_shared_by_n_other(
    eval_dataset: EvalDataset, n: int
) -> pd.DataFrame:
    # Generate df
    positives_series, _ = eval_dataset.get_predictions_for_positive_rate(
        desired_positive_rate=BEST_POS_RATE
    )

    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "y": eval_dataset.y,
            "y_pred": positives_series,
            "y_hat_probs": eval_dataset.y_hat_probs,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        }
    )

    df = df.dropna(subset=["outcome_timestamps"])

    df["outcome_uuid"] = df["id"].astype(str) + df["outcome_timestamps"].astype(str)

    # Count occurrences of each outcome_uuid
    outcome_uuid_counts = df["outcome_uuid"].value_counts()

    # Filter the DataFrame based on the count
    filtered_df = df[df["outcome_uuid"].isin(outcome_uuid_counts[outcome_uuid_counts == n].index)]

    return filtered_df


def _plot_model_outputs_over_time_for_prediction_times_with_outcome_shared_by_n_other(
    eval_dataset: EvalDataset, n: int
) -> pn.ggplot:
    df = _get_prediction_times_with_outcome_shared_by_n_other(eval_dataset, n)

    df["time_to_event"] = (df["outcome_timestamps"] - df["pred_timestamps"]).dt.days

    # plot a point for each prediction time, with time to event on the x-axis and the model output (y_hat_probs)$ on the y-axis. Colour the point red if y_pred is 0 and green if y_pred is 1. Connect points with the same outcome_uuid with a line.
    plot = (
        pn.ggplot(df)
        + FA_PN_THEME
        + pn.geom_point(
            pn.aes(x="time_to_event", y="y_hat_probs", color="y_pred"), size=2, alpha=0.5
        )
        + pn.geom_line(pn.aes(x="time_to_event", y="y_hat_probs", group="outcome_uuid"))
        + pn.labs(x="Time to event (days)", y="Model output")
        + pn.scale_color_manual(values=["#009E73", "#D55E00"])
        + pn.ggtitle(f"Outcomes with {n} prediction times")
    )

    return plot


def plot_model_outputs_over_time_for_cases_multiple_pred_times_per_outcome(
    run: ForcedAdmissionOutpatientPipelineRun,
    eval_dataset: EvalDataset,
    max_n: int,
    save: bool = True,
) -> pw.Bricks:
    plots = [
        _plot_model_outputs_over_time_for_prediction_times_with_outcome_shared_by_n_other(
            eval_dataset, n
        )
        for n in range(1, max_n)
    ]

    grid = create_patchwork_grid(plots=plots, single_plot_dimensions=(5, 5), n_in_row=2)

    if save:
        grid_output_path = (
            run.paper_outputs.paths.figures
            / "fa_outpatient_model_outputs_over_time_for_outcomes_with_multiple_pred_times.png"
        )
        grid.savefig(grid_output_path)

    return grid


def _plot_tpr_and_time_to_event_for_cases_wtih_n_pred_times_per_outcome(
    eval_dataset: EvalDataset, n: int
) -> pn.ggplot:
    df = _get_prediction_times_with_outcome_shared_by_n_other(eval_dataset, n)

    df["time_to_event"] = (df["outcome_timestamps"] - df["pred_timestamps"]).dt.days

    df["pred_time_order"] = (
        df.groupby("outcome_uuid")["pred_timestamps"].rank(method="first").astype(int)
    )

    if df["outcome_uuid"].nunique() < 5:
        return pn.ggplot()

    # Add true positive rate for each group
    df["tpr"] = ""

    for i in range(1, df["pred_time_order"].max() + 1):
        df.loc[df[df["pred_time_order"] == i].index, "tpr"] = (
            df[df["pred_time_order"] == i].y_pred.sum() / df[df["pred_time_order"] == i].y.sum()
        ) * 100

    # Change column dtypes
    df["tpr"] = pd.to_numeric(df["tpr"])
    df["pred_time_order"] = df["pred_time_order"].astype("category")

    # Create tpr for positining boxplot + violin plot away from each other:
    df["tpr_boxplot"] = df["tpr"] + 1.2
    df["tpr_violin"] = df["tpr"] - 0.6

    plot = (
        pn.ggplot(df)
        + pn.scale_x_continuous(limits=((df["tpr"].max() + 5), max((df["tpr"].min() - 5), 0)))
        + pn.scale_y_reverse()
        + FA_PN_THEME
        + pn.coord_flip()
        + pn.scale_fill_brewer(type="seq", palette="YlGn")  # type: ignore
        + pn.geom_point(
            pn.aes(x="tpr", y="time_to_event", fill="pred_time_order"), alpha=0.9, size=3
        )
        + pn.labs(x="Accuracy (%)", y="Time to event (days)")
        + pn.theme(legend_position="none")
        + pn.geom_violin(
            pn.aes(x="tpr_violin", y="time_to_event", fill="pred_time_order"),
            style="left",
            alpha=0.6,
            width=2,
            position=pn.position_dodge(),
        )
        + pn.geom_boxplot(
            pn.aes(x="tpr_boxplot", y="time_to_event", fill="pred_time_order"),
            show_legend=False,
            width=1,
            alpha=0.3,
            position=pn.position_dodge2(),
        )
        + pn.ggtitle(f"Outcomes with {n} prediction times")
    )

    return plot


def plot_distribution_of_n_pred_times_per_outcome(
    run: ForcedAdmissionOutpatientPipelineRun, eval_dataset: EvalDataset, max_n: int
) -> pn.ggplot:
    dist = [
        _get_prediction_times_with_outcome_shared_by_n_other(eval_dataset, n).outcome_uuid.nunique()
        for n in range(1, max_n)
    ]

    df = pd.DataFrame(
        {
            "n_pred_times": [
                category for category, count in zip(range(1, max_n), dist) for _ in range(count)
            ]
        }
    )

    n_pred_times_counts = df["n_pred_times"].value_counts()
    filtered_df = df[df["n_pred_times"].isin(n_pred_times_counts[n_pred_times_counts > 4].index)]
    filtered_df = filtered_df.reset_index(drop=True)

    plot = (
        pn.ggplot(filtered_df)
        + FA_PN_THEME
        + pn.geom_bar(pn.aes(x="n_pred_times"), fill="#009E73")
        + pn.labs(x="No. prediction times per outcome", y="Count")
        + pn.annotate(
            "text", x=7.5, y=60, label="Counts < 5 have been removed", size=10, color="red"
        )
    )

    auroc_path = run.paper_outputs.paths.figures / "fa_n_pred_times_per_outcome_distribution.png"
    plot.save(auroc_path)

    return plot


def plot_tpr_and_time_to_event_for_cases_wtih_multiple_pred_times_per_outcome(
    run: ForcedAdmissionOutpatientPipelineRun,
    eval_dataset: EvalDataset,
    max_n: int,
    save: bool = True,
) -> pw.Bricks:
    plots = [
        _plot_tpr_and_time_to_event_for_cases_wtih_n_pred_times_per_outcome(eval_dataset, n)
        for n in range(1, max_n)
    ]

    grid = create_patchwork_grid(plots=plots, single_plot_dimensions=(5, 5), n_in_row=2)

    if save:
        grid_output_path = (
            run.paper_outputs.paths.figures
            / "fa_outpatient_performance_and_distribution_for_outcomes_with_multiple_pred_times.png"
        )
        grid.savefig(grid_output_path)

    return grid


if __name__ == "__main__":
    from psycop.projects.forced_admission_outpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    run = get_best_eval_pipeline()
    eval_dataset = run.pipeline_outputs.get_eval_dataset()
    max_n = 7

    plot_model_outputs_over_time_for_cases_multiple_pred_times_per_outcome(run, eval_dataset, max_n)

    plot_tpr_and_time_to_event_for_cases_wtih_multiple_pred_times_per_outcome(
        run, eval_dataset, max_n
    )
