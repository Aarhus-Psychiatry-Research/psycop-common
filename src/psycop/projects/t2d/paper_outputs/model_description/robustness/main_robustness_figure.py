from psycop.projects.t2d.paper_outputs.config import BEST_EVAL_PIPELINE
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_age import (
    t2d_auroc_by_age,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_cyclic_time import (
    t2d_auroc_by_day_of_week,
    t2d_auroc_by_month_of_year,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_n_hba1c import (
    t2d_auroc_by_n_hba1c,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_sex import (
    t2d_auroc_by_sex,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_time_from_first_visit import (
    t2d_auroc_by_time_from_first_visit,
)
from psycop.projects.t2d.paper_outputs.utils.create_patchwork_figure import (
    t2d_create_patchwork_figure,
)
from psycop.projects.t2d.utils.best_runs import PipelineRun


def t2d_create_main_robustness_figure(run: PipelineRun) -> None:
    t2d_create_patchwork_figure(
        run=run,
        plot_fns=(
            t2d_auroc_by_sex,
            t2d_auroc_by_age,
            t2d_auroc_by_n_hba1c,
            t2d_auroc_by_time_from_first_visit,
            t2d_auroc_by_month_of_year,
            t2d_auroc_by_day_of_week,
        ),
        output_filename="t2d_main_robustness.png",
    )


if __name__ == "__main__":
    t2d_create_main_robustness_figure(run=BEST_EVAL_PIPELINE)
