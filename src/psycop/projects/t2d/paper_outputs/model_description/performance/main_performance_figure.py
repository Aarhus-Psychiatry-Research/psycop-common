from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.paper_outputs.create_patchwork_figure import (
    t2d_create_patchwork_figure,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.auroc import (
    t2d_auroc_plot,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.confusion_matrix_pipeline import (
    t2d_confusion_matrix_plot,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.incidence_by_time_until_diagnosis import (
    t2d_first_pred_to_event,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.sensitivity_by_time_to_event_pipeline import (
    t2d_sensitivity_by_time_to_event,
)
from psycop.projects.t2d.utils.best_runs import ModelRun


def t2d_create_main_performance_figure(run: ModelRun) -> None:
    t2d_create_patchwork_figure(
        run=run,
        plot_fns=(
            t2d_auroc_plot,
            t2d_confusion_matrix_plot,
            t2d_sensitivity_by_time_to_event,
            t2d_first_pred_to_event,
        ),
        output_filename="t2d_main_performance_figure.png",
    )


if __name__ == "__main__":
    t2d_create_main_performance_figure(run=EVAL_RUN)
