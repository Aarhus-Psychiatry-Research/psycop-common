import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_auroc_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.forced_admission_inpatient.model_eval.config import FA_PN_THEME
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def _plot_auroc_by_time_to_event(df: pd.DataFrame) -> pn.ggplot:
    p = (
        pn.ggplot(
            df,
            pn.aes(
                x="unit_from_event_binned",
                y="AUROC",
                ymin="ci_lower",
                ymax="ci_upper",
                color="actual_positive_rate",
            ),
        )
        + pn.scale_x_discrete()
        + pn.geom_point()
        + pn.geom_linerange(size=0.5)
        + pn.labs(x="Days to outcome", y="auroc")
        + FA_PN_THEME
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
        + pn.scale_color_brewer(type="qual", palette=2)
        + pn.labs(color="Predicted Positive Rate")
        + pn.theme(
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            legend_position=(0.3, 0.88),
        )
    )

    return p


def fa_inpatient_plot_auroc_by_time_to_event(df: pd.DataFrame) -> pn.ggplot:
    categories = df["unit_from_event_binned"].dtype.categories  # type: ignore
    df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
        new_categories=categories,
        ordered=True,  # type: ignore
    )

    p = _plot_auroc_by_time_to_event(df)

    return p


def auroc_by_time_to_event(eval_dataset: EvalDataset) -> pn.ggplot:
    if eval_dataset.outcome_timestamps is None:
        raise ValueError(
            "The outcome timestamps must be provided in order to calculate the auroc by time to event.",
        )

    df = get_auroc_by_timedelta_df(
        y=eval_dataset.y,  # type: ignore
        y_hat_probs=eval_dataset.y_hat_probs[0],
        time_one=eval_dataset.pred_timestamps,
        time_two=eval_dataset.outcome_timestamps,
        direction="t2-t1",
        bins=range(0, 180, 30),
        bin_unit="D",
        bin_continuous_input=True,
        drop_na_events=True,
    )

    p = fa_inpatient_plot_auroc_by_time_to_event(df)

    return p


def fa_inpatient_auroc_by_time_to_event(
    pipeline_run: ForcedAdmissionInpatientPipelineRun,
) -> pn.ggplot:
    eval_ds = pipeline_run.pipeline_outputs.get_eval_dataset()

    p = auroc_by_time_to_event(eval_dataset=eval_ds)

    p.save(
        pipeline_run.paper_outputs.paths.figures / "auroc_by_time_to_event.png",
        width=7,
        height=7,
    )

    return p


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_inpatient_auroc_by_time_to_event(pipeline_run=get_best_eval_pipeline())
