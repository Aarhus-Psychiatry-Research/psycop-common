import plotnine as pn

from psycop.common.model_evaluation.binary.global_performance.roc_auc import plot_auc_roc
from psycop.projects.clozapine.model_eval.selected_runs import get_best_eval_pipeline
from psycop.projects.clozapine.utils.pipeline_objects import ClozapinePipelineRun


def clozapine_auroc_plot(run: ClozapinePipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    p = plot_auc_roc(eval_dataset=eval_ds)

    auroc_path = run.paper_outputs.paths.figures / "clozapine_auroc_plot.png"
    p.save(auroc_path)

    return p


if __name__ == "__main__":
    clozapine_auroc_plot(run=get_best_eval_pipeline())
