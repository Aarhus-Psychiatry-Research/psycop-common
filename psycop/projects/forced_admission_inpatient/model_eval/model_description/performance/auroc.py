import plotnine as pn

from psycop.common.model_evaluation.binary.global_performance.roc_auc import (
    plot_auc_roc,
)
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    PipelineRun,
)


def fa_auroc_plot(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    p = plot_auc_roc(
        eval_dataset=eval_ds,
    )

    auroc_path = run.paper_outputs.paths.figures / "auc_roc.png"
    p.save(auroc_path)

    return p


if __name__ == "__main__":
    fa_auroc_plot(run=get_best_eval_pipeline)
