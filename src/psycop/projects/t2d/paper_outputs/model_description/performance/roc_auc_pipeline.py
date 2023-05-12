from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    plot_auc_roc,
)
from psycop.projects.t2d.paper_outputs.config import FIGURES_PATH
from psycop.projects.t2d.utils.best_runs import Run


def save_auroc_plot_for_t2d(run: Run):
    eval_ds = run.get_eval_dataset()

    auroc_path = FIGURES_PATH / "auc_roc.png"
    plot_auc_roc(
        eval_dataset=eval_ds,
        dpi=300,
        save_path=auroc_path,
    )

    print(f"Saving AUR-ROC plot to {auroc_path}")
