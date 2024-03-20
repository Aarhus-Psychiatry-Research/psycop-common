from psycop.projects.forced_admission_outpatient.model_eval.model_description.robustness.robustness_by_age import (
    fa_outpatient_auroc_by_age,
)
from psycop.projects.forced_admission_outpatient.model_eval.model_description.robustness.robustness_by_cyclic_time import (
    fa_outpatient_auroc_by_day_of_week,
    fa_outpatient_auroc_by_month_of_year,
)
from psycop.projects.forced_admission_outpatient.model_eval.model_description.robustness.robustness_by_sex import (
    fa_outpatient_auroc_by_sex,
)
from psycop.projects.forced_admission_outpatient.model_eval.model_description.robustness.robustness_by_time_from_first_visit import (
    fa_outpatient_auroc_by_time_from_first_visit,
)
from psycop.projects.forced_admission_outpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_outpatient.model_eval.utils.create_patchwork_figure import (
    fa_outpatient_create_patchwork_figure,
)
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def fa_outpatient_create_main_robustness_figure(run: ForcedAdmissionOutpatientPipelineRun) -> None:
    fa_outpatient_create_patchwork_figure(
        run=run,
        plot_fns=(
            fa_outpatient_auroc_by_sex,
            fa_outpatient_auroc_by_age,
            fa_outpatient_auroc_by_time_from_first_visit,
            fa_outpatient_auroc_by_month_of_year,
            fa_outpatient_auroc_by_day_of_week,
        ),
        output_filename=run.paper_outputs.artifact_names.main_robustness_figure,
        single_plot_dimensions=(5, 3),
    )


if __name__ == "__main__":
    fa_outpatient_create_main_robustness_figure(run=get_best_eval_pipeline())
