from psycop.common.model_evaluation.binary.subgroups.age import (
    plot_roc_auc_by_age,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, ROBUSTNESS_PATH
from psycop.projects.t2d.utils.best_runs import Run


def roc_auc_by_age(run: Run):
    print("Plotting AUC by age")
    eval_ds = run.get_eval_dataset()

    plot_roc_auc_by_age(
        eval_dataset=eval_ds,
        bins=[18, *range(20, 80, 10)],
        save_path=ROBUSTNESS_PATH / "auc_by_age.png",
    )


if __name__ == "__main__":
    roc_auc_by_age(run=EVAL_RUN)
