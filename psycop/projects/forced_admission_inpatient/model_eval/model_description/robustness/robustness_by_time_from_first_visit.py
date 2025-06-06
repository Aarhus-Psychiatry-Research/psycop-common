import plotnine as pn
import polars as pl

from psycop.common.model_evaluation.binary.time.timedelta_data import get_auroc_by_timedelta_df
from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.robustness_plot import (
    fa_inpatient_plot_robustness,
)
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def fa_inpatient_auroc_by_time_from_first_visit(
    run: ForcedAdmissionInpatientPipelineRun,
) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = pl.DataFrame(
        {
            "id": eval_ds.ids,
            "y": eval_ds.y,
            "y_hat_probs": eval_ds.y_hat_probs,
            "pred_timestamp": eval_ds.pred_timestamps,
        }
    )

    # @Martin - correct me if I'm wrong, but isn't this rather the first prediction
    # time than the first visit?
    first_visit = (
        df.sort("pred_timestamp", descending=False)
        .groupby("id")
        .head(1)
        .rename({"pred_timestamp": "first_visit_timestamp"})
    )

    df = df.join(first_visit.select(["first_visit_timestamp", "id"]), on="id").to_pandas()

    plot_df = get_auroc_by_timedelta_df(
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
        time_one=df["first_visit_timestamp"],
        time_two=df["pred_timestamp"],
        direction="t2-t1",
        bin_unit="M",
        bins=range(0, 60, 6),
    )

    return fa_inpatient_plot_robustness(
        plot_df,
        x_column="unit_from_event_binned",
        line_y_col_name="auroc",
        xlab="Months since first visit to Psychiatric Services",
    )


if __name__ == "__main__":
    fa_inpatient_auroc_by_time_from_first_visit(run=get_best_eval_pipeline())
