from psycop.common.model_evaluation.binary.time.periodic_plots import (
    plot_roc_auc_by_periodic_time,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, ROBUSTNESS_PATH
from psycop.projects.t2d.utils.best_runs import Run


def roc_auc_by_hour_of_day(run: Run):
    eval_ds = run.get_eval_dataset()
    plot_roc_auc_by_periodic_time(
        eval_dataset=eval_ds,
        bin_period="H",
        save_path=ROBUSTNESS_PATH / "auc_by_hour_of_day.png",
    )


def auroc_by_day_of_week(run: Run):
    eval_ds = run.get_eval_dataset()

    plot_roc_auc_by_periodic_time(
        eval_dataset=eval_ds,
        bin_period="D",
        save_path=ROBUSTNESS_PATH / "auc_by_day_of_week.png",
    )


def auroc_by_month_of_year(run: Run):
    eval_ds = run.get_eval_dataset()

    plot_roc_auc_by_periodic_time(
        eval_dataset=eval_ds,
        bin_period="M",
        save_path=ROBUSTNESS_PATH / "auc_by_month_of_year.png",
    )


def roc_auc_by_cyclic_time():
    print("Plotting AUC by cyclic time")
    EVAL_RUN.get_eval_dataset()


if __name__ == "__main__":
    roc_auc_by_cyclic_time()
