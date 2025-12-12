import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.model_description.robustness.robustness_plot import (
    fa_outpatient_plot_robustness,
)
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_outpatient.old_project_code.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def fa_outpatient_auroc_by_sex(run: ForcedAdmissionOutpatientPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset(custom_columns=["is_female"])

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.is_female,  # type: ignore
        input_name="is_female",
        confidence_interval=True,
        bin_continuous_input=False,
    )

    int_to_sex = {0: "Male", 1: "Female"}
    df["Sex"] = df["is_female"].map(int_to_sex).astype("category")

    return fa_outpatient_plot_robustness(df, x_column="Sex", line_y_col_name="auroc", xlab="Sex")


if __name__ == "__main__":
    fa_outpatient_auroc_by_sex(run=get_best_eval_pipeline())
