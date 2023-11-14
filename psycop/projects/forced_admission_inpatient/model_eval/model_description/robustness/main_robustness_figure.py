from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.robustness_by_age import (
    fa_inpatient_auroc_by_age,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.robustness_by_cyclic_time import (
    fa_inpatient_auroc_by_day_of_week,
    fa_inpatient_auroc_by_month_of_year,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.robustness_by_n_hba1c import (
    fa_inpatient_auroc_by_n_hba1c,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.robustness_by_sex import (
    fa_inpatient_auroc_by_sex,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.robustness.robustness_by_time_from_first_visit import (
    fa_inpatient_auroc_by_time_from_first_visit,
)
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.model_eval.utils.create_patchwork_figure import (
    fa_inpatient_create_patchwork_figure,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def fa_inpatient_create_main_robustness_figure(
    run: ForcedAdmissionInpatientPipelineRun,
) -> None:
    fa_inpatient_create_patchwork_figure(
        run=run,
        plot_fns=(
            fa_inpatient_auroc_by_sex,
            fa_inpatient_auroc_by_age,
            fa_inpatient_auroc_by_n_hba1c,
            fa_inpatient_auroc_by_time_from_first_visit,
            fa_inpatient_auroc_by_month_of_year,
            fa_inpatient_auroc_by_day_of_week,
        ),
        output_filename="main_robustness_figure.png",
        single_plot_dimensions=(5, 3),
    )


if __name__ == "__main__":
    fa_inpatient_create_main_robustness_figure(run=get_best_eval_pipeline())
