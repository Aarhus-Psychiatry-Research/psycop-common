import pandas as pd
import plotnine as pn
from psycop.common.model_evaluation.confusion_matrix.confusion_matrix import (
    get_confusion_matrix_cells_from_df,
)
from psycop.projects.t2d.paper_outputs.model_description.performance.plotnine_confusion_matrix import (
    plotnine_confusion_matrix,
)
from psycop.projects.t2d.utils.best_runs import ModelRun


def t2d_confusion_matrix_plot(run: ModelRun) -> pn.ggplot:
    eval_ds = run.get_eval_dataset()

    df = pd.DataFrame(
        {
            "true": eval_ds.y,
            "pred": eval_ds.get_predictions_for_positive_rate(run.pos_rate),
        }
    )
    confusion_matrix = get_confusion_matrix_cells_from_df(df=df)

    return plotnine_confusion_matrix(
        matrix=confusion_matrix,
        x_title=f"T2D within {int(run.cfg.preprocessing.pre_split.min_lookahead_days/365.25)} years",
    )
