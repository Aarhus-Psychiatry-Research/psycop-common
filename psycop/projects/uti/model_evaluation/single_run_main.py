# Main script for generating main evaluation figure (AUROC plot and confusion matrix)
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import patchworklib as pw
import polars as pl

from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk
from psycop.projects.uti.model_evaluation.auroc.model import auroc_model
from psycop.projects.uti.model_evaluation.auroc.view import AUROCPlot
from psycop.projects.uti.model_evaluation.confusion_matrix.model import confusion_matrix_model
from psycop.projects.uti.model_evaluation.confusion_matrix.view import ConfusionMatrixPlot
from psycop.projects.uti.model_evaluation.single_run_artifact import SingleRunPlot

if TYPE_CHECKING:
    from collections.abc import Sequence

    import plotnine as pn

log = logging.getLogger(__name__)


def single_run_main(
    eval_df: pl.DataFrame, desired_positive_rate: float, outcome_label: str, first_letter_index: int
) -> pw.Bricks:
    plots: Sequence[SingleRunPlot] = [
        AUROCPlot(auroc_model(eval_df=eval_df, n_bootstraps=5)),
        ConfusionMatrixPlot(
            confusion_matrix_model(eval_df=eval_df, desired_positive_rate=desired_positive_rate),
            outcome_label=outcome_label,
        ),
    ]

    ggplots: list[pn.ggplot] = []
    for plot in plots:
        log.info(f"Starting processing of {plot.__class__.__name__}")
        ggplots.append(plot())

    figure = create_patchwork_grid(
        plots=ggplots,
        single_plot_dimensions=(5, 4.5),
        n_in_row=2,
        first_letter_index=first_letter_index,
    )
    return figure


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    MAIN_METRIC = "all_oof_BinaryAUROC"

    experiment = "uti_hparam_test_run"
    experiment_path = f"E:/shared_resources/uti/eval_runs/{experiment}_best_run_evaluated_on_test"
    experiment_df = read_eval_df_from_disk(experiment_path)

    save_dir = Path(experiment_path + "/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    figure = single_run_main(
        eval_df=experiment_df, desired_positive_rate=0.02, outcome_label="UTI", first_letter_index=0
    )

    figure.savefig(save_dir / "uti_main_plot.png")
