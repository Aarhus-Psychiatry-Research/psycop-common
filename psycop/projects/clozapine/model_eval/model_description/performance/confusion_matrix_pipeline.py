import pandas as pd
import plotnine as pn

from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)
from psycop.projects.clozapine.model_eval.model_description.performance.plotnine_confusion_matrix import (
    plotnine_confusion_matrix,
)
from psycop.projects.clozapine.model_eval.selected_runs import get_best_eval_pipeline
from psycop.projects.clozapine.utils.pipeline_objects import ClozapinePipelineRun


def clozapine_confusion_matrix_plot(run: ClozapinePipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = pd.DataFrame(
        {
            "true": eval_ds.y,
            "pred": eval_ds.get_predictions_for_positive_rate(run.paper_outputs.pos_rate)[0],
        }
    )
    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)

    p = plotnine_confusion_matrix(
        matrix=confusion_matrix,
        outcome_text=f"FAs within {int(run.inputs.cfg.preprocessing.pre_split.min_lookahead_days)} days",
    )

    p.save(run.paper_outputs.paths.figures / "clozapine_confusion_matrix.png")

    return p


if __name__ == "__main__":
    clozapine_confusion_matrix_plot(run=get_best_eval_pipeline())
