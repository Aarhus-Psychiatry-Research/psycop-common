import plotnine as pn

from psycop.common.model_evaluation.binary.time.periodic_data import roc_auc_by_periodic_time_df
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.model_description.robustness.robustness_plot import (
    fa_outpatient_plot_robustness,
)
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_outpatient.old_project_code.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def fa_outpatient_auroc_by_day_of_week(run: ForcedAdmissionOutpatientPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="D",
    )

    return fa_outpatient_plot_robustness(
        df, x_column="time_bin", line_y_col_name="auroc", xlab="Day of Week"
    )


def fa_outpatient_auroc_by_month_of_year(run: ForcedAdmissionOutpatientPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = roc_auc_by_periodic_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="M",
    )

    return fa_outpatient_plot_robustness(
        df, x_column="time_bin", line_y_col_name="auroc", xlab="Month of Year"
    )


if __name__ == "__main__":
    fa_outpatient_auroc_by_day_of_week(run=get_best_eval_pipeline())
    fa_outpatient_auroc_by_month_of_year(run=get_best_eval_pipeline())
