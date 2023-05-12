from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    plot_auc_roc,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH


def roc_auc_pipeline():
    eval_ds = EVAL_RUN.get_eval_dataset()

    plot_auc_roc(
        eval_dataset=eval_ds,
        dpi=300,
        save_path=FIGURES_PATH / "auc_roc.png",
    )


if __name__ == "__main__":
    roc_auc_pipeline()
