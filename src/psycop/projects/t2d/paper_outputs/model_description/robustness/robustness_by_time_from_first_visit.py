from psycop.common.model_evaluation.binary.time.timedelta_plots import (
    plot_roc_auc_by_time_from_first_visit,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, ROBUSTNESS_PATH


def roc_auc_by_time_from_first_visit():
    print("Plotting AUC by time from first visit")
    eval_ds = EVAL_RUN.get_eval_dataset()

    plot_roc_auc_by_time_from_first_visit(
        eval_dataset=eval_ds,
        bins=range(0, 37, 3),
        bin_unit="M",
        save_path=ROBUSTNESS_PATH / "auc_by_time_from_first_visit.png",
    )


if __name__ == "__main__":
    roc_auc_by_time_from_first_visit()
