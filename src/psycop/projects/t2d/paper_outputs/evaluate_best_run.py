import datetime

import pandas as pd
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.paper_outputs.model_description.performance.confusion_matrix_pipeline import (
    confusion_matrix_pipeline,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.incidence_by_time_until_diagnosis import (
    incidence_by_time_until_outcome_pipeline,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.performance_by_ppr import (
    output_performance_by_ppr,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.roc_auc_pipeline import (
    save_auroc_plot_for_t2d,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.sensitivity_by_time_to_event_pipeline import (
    sensitivity_by_time_to_event,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_age import (
    roc_auc_by_age,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_cyclic_time import (
    auroc_by_day_of_week,
    auroc_by_month_of_year,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_n_hba1c import (
    plot_auroc_by_n_hba1c,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_sex import (
    roc_auc_by_sex,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_time_from_first_visit import (
    roc_auc_by_time_from_first_visit,
)
from psycop.projects.t2d.utils.best_runs import Run
from wasabi import Printer

pd.set_option("mode.chained_assignment", None)


def evaluate_best_run(run: Run):
    msg = Printer(timestamp=True)
    msg.info(f"Evaluating {run.name}")

    for output_str, value in (
        ("Run group", run.group.name),
        ("Model_type", run.model_type),
        ("Lookahead days", run.cfg.preprocessing.pre_split.min_lookahead_days),
    ):
        msg.info(f"    {output_str}: {value}")

    output_fns = {
        "performance_figures": [
            save_auroc_plot_for_t2d,
            confusion_matrix_pipeline,
            incidence_by_time_until_outcome_pipeline,
            sensitivity_by_time_to_event,
        ],
        "robustness": [
            roc_auc_by_sex,
            roc_auc_by_age,
            plot_auroc_by_n_hba1c,
            roc_auc_by_time_from_first_visit,
            auroc_by_month_of_year,
            auroc_by_day_of_week,
        ],
        "performance_by_ppr_table": [output_performance_by_ppr],
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
