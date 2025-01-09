from psycop.projects.t2d_extended.model_evaluation.model_description.performance.auroc import t2d_extended_auroc_plot
from psycop.projects.t2d_extended.model_evaluation.model_description.performance.confusion_matrix_pipeline import (
    t2d_extended_confusion_matrix_plot,
)
from psycop.projects.t2d_extended.model_evaluation.model_description.performance.incidence_by_time_until_diagnosis import (
    t2d_extended_first_pred_to_event,
)
from psycop.projects.t2d_extended.model_evaluation.model_description.performance.sensitivity_by_time_to_event_pipeline import (
    t2d_extended_sensitivity_by_time_to_event,
)
from psycop.projects.t2d_extended.model_evaluation.utils.create_patchwork_figure import (
    t2d_extended_create_patchwork_figure,
)
from psycop.projects.t2d_extended.utils.pipeline_objects import T2DExtendedPipelineRun


def t2d_extended_create_main_performance_figure(run: T2DExtendedPipelineRun) -> None:
    t2d_extended_create_patchwork_figure(
        run=run,
        plot_fns=(
            t2d_extended_auroc_plot,
            t2d_extended_confusion_matrix_plot,
            t2d_extended_sensitivity_by_time_to_event,
            t2d_extended_first_pred_to_event,
        ),
        output_filename=run.paper_outputs.artifact_names.main_performance_figure,
    )


if __name__ == "__main__":
    from psycop.projects.t2d_extended.model_evaluation.selected_runs import get_best_eval_pipeline

    t2d_extended_create_main_performance_figure(run=get_best_eval_pipeline())
