from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.utils.best_runs import Run


def roc_auc_by_time_from_first_visit(run: Run):
    print("Plotting AUC by time from first visit")
    run.get_eval_dataset()

    # TODO: Create plot


if __name__ == "__main__":
    roc_auc_by_time_from_first_visit(run=EVAL_RUN)
