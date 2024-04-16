import pandas as pd
import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import (
    scz_bp_get_eval_ds_from_best_run_in_experiment,
)


def _plot_metric_by_time_to_event(
    df: pd.DataFrame, metric: str, plot_combined: bool = False
) -> pn.ggplot:
    df["subset"] = df["subset"].replace({"bp": "BP", "scz": "SCZ", "both": "Combined"})
    df["subset"] = pd.Categorical(df["subset"], ["BP", "SCZ", "Combined"])
    if not plot_combined:
        df = df.query("subset != 'Combined'").copy()
        df["subset"] = pd.Categorical(df["subset"], ["BP", "SCZ"])

    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="unit_from_event_binned",
                y=metric,
                ymin="ci_lower",
                ymax="ci_upper",
                color="subset",
                group="subset",
            ),
        )
        + pn.scale_x_discrete(reverse=True)
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.geom_line()
        + pn.scale_color_manual(values=["#669BBC", "#A8C686", "#F3A712"])
        + pn.theme_minimal()
        + pn.theme(
            legend_position=(0.3, 0.92),
            axis_text_x=pn.element_text(rotation=45, hjust=1),
            axis_title=pn.element_text(size=14),
            legend_title=pn.element_blank(),
            legend_text=pn.element_text(size=11),
            axis_text=pn.element_text(size=10),
            figure_size=(5, 5),
        )
    )

    return p


def reverse_x_axis_categories(df: pd.DataFrame) -> pd.DataFrame:
    categories = df["unit_from_event_binned"].dtype.categories[::-1]  # type: ignore
    df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
        new_categories=categories,
        ordered=True,  # type: ignore
    )
    return df


def scz_bp_get_sensitivity_by_time_to_event_df(eval_ds: EvalDataset, ppr: float) -> pd.DataFrame:
    dfs = []

    if eval_ds.outcome_timestamps is None:
        raise ValueError(
            "The outcome timestamps must be provided in order to calculate the metric by time to event."
        )

    eval_dataset_polars = eval_ds.to_polars()
    eval_dataset_df = eval_dataset_polars.with_columns(
        pl.Series(
            name="y_hat",
            values=eval_ds.get_predictions_for_positive_rate(desired_positive_rate=ppr)[0],
        )
    )
    for diagnosis_subset in ["scz", "bp", "both"]:
        if diagnosis_subset != "both":
            filtered_df = eval_dataset_df.filter(pl.col("scz_or_bp") == diagnosis_subset)
        else:
            filtered_df = eval_dataset_df

        df = get_sensitivity_by_timedelta_df(
            y=filtered_df["y"],
            y_hat=filtered_df["y_hat"],
            time_one=filtered_df["pred_timestamps"],
            time_two=filtered_df["outcome_timestamps"],
            direction="t2-t1",
            bins=range(0, 60, 6),
            bin_unit="M",
            bin_continuous_input=True,
            drop_na_events=True,
        )
        # Convert to string to allow distinct scales for color
        df["actual_positive_rate"] = str(ppr)
        df["subset"] = diagnosis_subset
        dfs.append(df)

    return pd.concat(dfs)


def scz_bp_plot_sensitivity_by_time_to_event(eval_ds: EvalDataset, ppr: float) -> pn.ggplot:
    plot_df = scz_bp_get_sensitivity_by_time_to_event_df(eval_ds=eval_ds, ppr=ppr)
    plot_df = reverse_x_axis_categories(plot_df)

    p = _plot_metric_by_time_to_event(df=plot_df, metric="sensitivity") + pn.labs(
        x="Months to outcome", y="Sensitivitiy", color="Predicted Positive Rate"
    )

    return p


if __name__ == "__main__":
    best_experiment = "sczbp/text_only"
    best_pos_rate = 0.04
    eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=best_experiment)

    p = scz_bp_plot_sensitivity_by_time_to_event(eval_ds=eval_ds, ppr=best_pos_rate)
    p + pn.theme(
        legend_position=(0.4, 0.92),
        axis_text_x=pn.element_text(rotation=45, hjust=1),
        axis_title=pn.element_text(size=14),
        legend_title=pn.element_blank(),
        legend_text=pn.element_text(size=11),
        axis_text=pn.element_text(size=10),
        figure_size=(5, 5),
    )
