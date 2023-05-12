from psycop.common.model_evaluation.binary.global_performance.precision_recall import (
    plot_precision_recall,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH


def precision_recall_pipeline():
    eval_ds = EVAL_RUN.get_eval_dataset()

    plot_precision_recall(
        eval_dataset=eval_ds,
        title="Precision-recall curve",
        save_path=FIGURES_PATH / "precision_recall.png",
    )


if __name__ == "__main__":
    precision_recall_pipeline()
