from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.auroc import (
    fa_inpatient_auroc_plot,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.confusion_matrix_pipeline import (
    fa_inpatient_confusion_matrix_plot,
)
from psycop.projects.forced_admission_inpatient.model_eval.model_description.performance.sensitivity_by_time_to_event import (
    fa_inpatient_sensitivity_by_time_to_event,
)
from psycop.projects.forced_admission_inpatient.model_eval.utils.create_patchwork_figure import (
    fa_inpatient_create_patchwork_figure,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def fa_inpatient_create_main_performance_figure(
    run: ForcedAdmissionInpatientPipelineRun,
) -> None:
    fa_inpatient_create_patchwork_figure(
        run=run,
        plot_fns=(
            fa_inpatient_auroc_plot,
            fa_inpatient_confusion_matrix_plot,
            fa_inpatient_sensitivity_by_time_to_event,
        ),
        output_filename="main_performance_figure.png",
    )


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_inpatient_create_main_performance_figure(run=get_best_eval_pipeline())
