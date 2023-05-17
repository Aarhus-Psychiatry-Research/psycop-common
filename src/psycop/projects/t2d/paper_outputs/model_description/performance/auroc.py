import plotnine as pn
from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    plot_auc_roc,
)
from psycop.projects.t2d.paper_outputs.config import FIGURES_PATH
from psycop.projects.t2d.utils.best_runs import ModelRun


def t2d_auroc_plot(run: ModelRun) -> pn.ggplot:
    eval_ds = run.get_eval_dataset()

    return plot_auc_roc(
        eval_dataset=eval_ds,
    )


if __name__ == "__main__":
    auroc_path = FIGURES_PATH / "auc_roc.png"
    t2d_auroc_plot.save(auroc_path)
    print(f"Saving AUR-ROC plot to {auroc_path}")
