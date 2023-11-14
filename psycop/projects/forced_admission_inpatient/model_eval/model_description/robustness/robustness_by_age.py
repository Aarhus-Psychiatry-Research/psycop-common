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


def fa_inpatient_auroc_by_age(run: ForcedAdmissionInpatientPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 10)],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    p = fa_inpatient_plot_robustness(
        df,
        x_column="age_binned",
        line_y_col_name="auroc",
        xlab="Age",
        rotate_x_axis_labels_degrees=45,
    )

    return p


if __name__ == "__main__":
    fa_inpatient_auroc_by_age(run=get_best_eval_pipeline())
