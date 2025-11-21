from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.model_description.performance.auroc import (
    fa_outpatient_auroc_plot,
)
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.model_description.performance.confusion_matrix_pipeline import (
    fa_outpatient_confusion_matrix_plot,
)
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.model_description.performance.sensitivity_by_time_to_event import (
    fa_outpatient_sensitivity_by_time_to_event,
)
from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.utils.create_patchwork_figure import (
    fa_outpatient_create_patchwork_figure,
)
from psycop.projects.forced_admission_outpatient.old_project_code.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)


def fa_outpatient_create_main_performance_figure(run: ForcedAdmissionOutpatientPipelineRun) -> None:
    fa_outpatient_create_patchwork_figure(
        run=run,
        plot_fns=(
            fa_outpatient_auroc_plot,
            fa_outpatient_confusion_matrix_plot,
            fa_outpatient_sensitivity_by_time_to_event,
        ),
        output_filename=run.paper_outputs.artifact_names.main_performance_figure,
        single_plot_dimensions=(5, 3),
    )


if __name__ == "__main__":
    from psycop.projects.forced_admission_outpatient.old_project_code.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_outpatient_create_main_performance_figure(run=get_best_eval_pipeline())
