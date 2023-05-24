import datetime

import pandas as pd
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.paper_outputs.model_description.performance.auroc import (
    t2d_auroc_plot,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.performance_by_ppr import (
    t2d_output_performance_by_ppr,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.sensitivity_by_time_to_event_pipeline import (
    t2d_sensitivity_by_time_to_event,
)
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
from psycop.projects.t2d.utils.best_runs import PipelineRun
from wasabi import Printer

pd.set_option("mode.chained_assignment", None)


def evaluate_best_run(run: PipelineRun):
    msg = Printer(timestamp=True)
    msg.info(f"Evaluating {run.name}")

    output_fns = {
        "performance_figures": [
            t2d_auroc_plot,
            t2d_sensitivity_by_time_to_event,
        ],
        "robustness": [
            t2d_auroc_by_sex,
            t2d_auroc_by_age,
            t2d_auroc_by_n_hba1c,
            t2d_auroc_by_time_from_first_visit,
            t2d_auroc_by_month_of_year,
            t2d_auroc_by_day_of_week,
        ],
        "performance_by_ppr_table": [t2d_output_performance_by_ppr],
    }

    for group in output_fns:
        datetime.datetime.now()

        for fn in output_fns[group]:
            try:
                now = datetime.datetime.now()
                fn(run)
                finished = datetime.datetime.now()
                msg.good(
                    f"{fn.__name__} finished in {round((finished - now).seconds, 0)} seconds",
                )
            except Exception:
                msg.fail(f"{fn.__name__} failed")
        datetime.datetime.now()


if __name__ == "__main__":
    evaluate_best_run(run=EVAL_RUN)
