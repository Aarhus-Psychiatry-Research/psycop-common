import pandas as pd
import plotnine as pn
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.plotnine_confusion_matrix import (
    plotnine_confusion_matrix,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun


def t2d_confusion_matrix_plot(run: PipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    df = pd.DataFrame(
        {
            "true": eval_ds.y,
            "pred": eval_ds.get_predictions_for_positive_rate(
                run.paper_outputs.pos_rate,
            )[0],
        },
    )
    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)

    p = plotnine_confusion_matrix(
        matrix=confusion_matrix,
        x_title=f"T2D within {int(run.inputs.cfg.preprocessing.pre_split.min_lookahead_days/365)} years",
    )

    p.save(run.paper_outputs.paths.figures / "t2d_confusion_matrix_plot.png")

    return p


if __name__ == "__main__":
    t2d_confusion_matrix_plot(run=BEST_EVAL_PIPELINE)
