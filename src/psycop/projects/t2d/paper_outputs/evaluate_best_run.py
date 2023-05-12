import datetime

from psycop.common.model_evaluation.binary.subgroups.age import plot_roc_auc_by_age
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.paper_outputs.model_description.performance.confusion_matrix_pipeline import (
    confusion_matrix_pipeline,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.performance_by_ppr import (
    output_performance_by_ppr,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.roc_auc_pipeline import (
    save_auroc_plot_for_t2d,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.sensitivity_by_time_to_event_pipeline import (
    incidence_by_time_until_outcome_pipeline,
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


def evaluate_best_run(run: Run):
    output_fns = {
        "performance_figures": [
            save_auroc_plot_for_t2d,
            confusion_matrix_pipeline,
            incidence_by_time_until_outcome_pipeline,
            incidence_by_time_until_outcome_pipeline,
        ],
        "robustness": [
            roc_auc_by_sex,
            plot_roc_auc_by_age,
            plot_auroc_by_n_hba1c,
            roc_auc_by_time_from_first_visit,
            auroc_by_month_of_year,
            auroc_by_day_of_week,
        ],
        "performance_by_ppr_table": [output_performance_by_ppr],
    }

    for group in output_fns:
        group_start = datetime.datetime.now()

        for fn in output_fns[group]:
            try:
                now = datetime.datetime.now()
                fn(run)
                finished = datetime.datetime.now()
                print(f"Finished {fn.__name__} in {round((finished - now).seconds, 0)}")
            except Exception as e:
                print(f"Failed to run {fn.__name__} with error: {e}")

        group_finished = datetime.datetime.now()
        print(f"Finished group in {round((group_finished - group_start).seconds, 0)}")


if __name__ == "__main__":
    evaluate_best_run(run=EVAL_RUN)
