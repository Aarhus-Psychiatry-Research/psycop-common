import plotnine as pn
from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    plot_auc_roc,
)
from psycop.projects.t2d.paper_outputs.config import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.best_runs import PipelineRun


def t2d_auroc_plot(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    p = plot_auc_roc(
        eval_dataset=eval_ds,
    )

    auroc_path = run.paper_outputs.paths.figures / "auc_roc.png"
    p.save(auroc_path)

    return p


if __name__ == "__main__":
    t2d_auroc_plot(run=BEST_EVAL_PIPELINE)
