from datetime import datetime

from psycop.common.model_evaluation.patchwork.patchwork_grid import (
    create_patchwork_grid,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH
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
from wasabi import Printer

msg = Printer(timestamp=True)


def create_full_performance_figure(run: ModelRun):
    for output_str, value in (
        ("Run group", run.group.name),
        ("Model_type", run.model_type),
        ("Lookahead days", run.cfg.preprocessing.pre_split.min_lookahead_days),
    ):
        msg.info(f"    {output_str}: {value}")

    plot_fns = [
        t2d_auroc_plot,
        t2d_confusion_matrix_plot,
        t2d_sensitivity_by_time_to_event,
        t2d_first_pred_to_event,
    ]

    plots = []

    any_plot_failed = False

    for fn in plot_fns:
        try:
            now = datetime.now()
            p = fn(run)
            plots.append(p)
            p.save(FIGURES_PATH / f"{fn.__name__}.png")
            finished = datetime.now()

            msg.good(
                f"{fn.__name__} finished in {round((finished - now).seconds, 0)} seconds",
            )
        except Exception:
            msg.fail(f"{fn.__name__} failed")
            any_plot_failed = True
    datetime.now()

    if not any_plot_failed:
        grid = create_patchwork_grid(
            plots=plots,
            single_plot_dimensions=(7, 5),
            n_in_row=2,
        )
        grid.savefig(FIGURES_PATH / "full_performance_figure.png")


if __name__ == "__main__":
    create_full_performance_figure(run=EVAL_RUN)
