import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_CUSTOM_COLUMNS
from psycop.projects.t2d.paper_outputs.config import (
    T2D_PN_THEME,
)
from psycop.projects.scz_bp.evaluation.pipeline_objects import PipelineRun
import polars as pl

def _plot_metric_by_time_to_event(df: pd.DataFrame, metric: str) -> pn.ggplot:
    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="unit_from_event_binned",
                y=metric,
                ymin="ci_lower",
                ymax="ci_upper",
                color="actual_positive_rate",
            ),
        )
        + pn.scale_x_discrete(reverse=True)
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.scale_color_brewer(type="qual", palette=2)
        + T2D_PN_THEME
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.3, 0.88),
            axis_text_x=pn.element_text(rotation=45, hjust=1)
        )
        + pn.facet_wrap("~ subset")
        # + pn.labs(x="Months to outcome", y="Sensitivity")
        # + pn.labs(color="Predicted Positive Rate")
    )

    for value in df["actual_positive_rate"].unique():
        p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)

    return p


def reverse_x_axis_categories(df: pd.DataFrame) -> pd.DataFrame:
    categories = df["unit_from_event_binned"].dtype.categories[::-1]  # type: ignore
    df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
        new_categories=categories,
        ordered=True,  # type: ignore
    )
    return df

def scz_bp_get_sensitivity_by_time_to_event_df(eval_dataset: EvalDataset) -> pd.DataFrame:
    dfs = []

    if eval_dataset.outcome_timestamps is None:
        raise ValueError(
            "The outcome timestamps must be provided in order to calculate the metric by time to event.",
        )


    for ppr in [0.01, 0.03, 0.05]:
        eval_dataset_df = eval_dataset.to_polars().with_columns(pl.Series(name="y_hat", values=eval_dataset.get_predictions_for_positive_rate(desired_positive_rate=ppr)[0]))
        for diagnosis_subset in ["scz", "bp", "both"]:
            if diagnosis_subset != "both":
                filtered_df = eval_dataset_df.filter(pl.col("meta_scz_or_bp_indicator") == diagnosis_subset)
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


def scz_bp_plot_sensitivity_by_time_to_event(eval_dataset: EvalDataset) -> pn.ggplot:
    plot_df = scz_bp_get_sensitivity_by_time_to_event_df(eval_dataset=eval_dataset)
    plot_df = reverse_x_axis_categories(plot_df)
    
    p = (_plot_metric_by_time_to_event(df=plot_df, metric="sensitivity")
    + pn.labs(x="Months to outcome", y="Sensitivitiy", color="Predicted Positive Rate")
    )
    
    return p


def scz_bp_sensitivity_by_time_to_event(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset(custom_columns=SCZ_BP_CUSTOM_COLUMNS)
    p = scz_bp_plot_sensitivity_by_time_to_event(eval_dataset=eval_ds)

    p.save(run.paper_outputs.paths.figures / "recall_by_time_to_event", width=7, height=7)

    return p


if __name__ == "__main__":
    from psycop.projects.scz_bp.evaluation.model_selection.performance_by_group_lookahead_model_type import (
        DEVELOPMENT_PIPELINE_RUN,
    )
    # TODO make separately for scz/bp OR add as shapes

    p = scz_bp_sensitivity_by_time_to_event(run=DEVELOPMENT_PIPELINE_RUN)
