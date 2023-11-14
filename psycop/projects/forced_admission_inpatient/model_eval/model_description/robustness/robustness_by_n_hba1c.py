import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.robustness_plot import (
    fa_inpatient_plot_robustness,
)
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def fa_inpatient_auroc_by_n_hba1c(
    run: ForcedAdmissionInpatientPipelineRun,
) -> pn.ggplot:
    """Plot performance by n hba1c"""
    eval_ds = run.pipeline_outputs.get_eval_dataset(
        custom_columns=["eval_hba1c_within_9999_days_count_fallback_nan"],
    )

    col_name = "eval_hba1c_within_9999_days_count_fallback_nan"
    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.custom_columns[col_name],  # type: ignore
        input_name=col_name,
        bins=[0, 2, 4, 6, 8, 10, 12],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    return fa_inpatient_plot_robustness(
        df,
        x_column="eval_hba1c_within_9999_days_count_fallback_nan_binned",
        line_y_col_name="auroc",
        xlab="n HbA1c measurements prior to visit",
    )


if __name__ == "__main__":
    fa_inpatient_auroc_by_n_hba1c(run=get_best_eval_pipeline())
