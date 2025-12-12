import plotnine as pn

from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.model_description.robustness.robustness_plot import (
    fa_outpatient_plot_robustness,
)
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_outpatient.old_project_code.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def fa_outpatient_auroc_by_quarter(run: ForcedAdmissionOutpatientPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = create_roc_auc_by_absolute_time_df(
        labels=eval_ds.y,  # type: ignore
        y_hat_probs=eval_ds.y_hat_probs,  # type: ignore
        timestamps=eval_ds.pred_timestamps,
        bin_period="Q",
        confidence_interval=True,
    )

    return fa_outpatient_plot_robustness(
        df, x_column="time_bin", line_y_col_name="auroc", xlab="Quarter"
    )


if __name__ == "__main__":
    fa_outpatient_auroc_by_quarter(run=get_best_eval_pipeline())
