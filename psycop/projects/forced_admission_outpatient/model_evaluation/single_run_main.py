import logging
from pathlib import Path
from typing import TYPE_CHECKING

import patchworklib as pw
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.roc_by_multiple_runs_model import (
    ExperimentWithNames,
    group_auroc_model,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.auroc_by.roc_by_multiple_runs_view import (
    ROCByGroupPlot,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.confusion_matrix.model import (
    confusion_matrix_model,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.confusion_matrix.view import (
    ConfusionMatrixPlot,
)
from psycop.projects.forced_admission_outpatient.model_evaluation.single_run_artifact import (
    SingleRunPlot,
)
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk

if TYPE_CHECKING:
    from collections.abc import Sequence

    import plotnine as pn

log = logging.getLogger(__name__)


def single_run_main(
    eval_df: pl.DataFrame,
    group_auroc_experiments: ExperimentWithNames,
    desired_positive_rate: float,
    outcome_label: str,
    first_letter_index: int,
) -> pw.Bricks:
    eval_df = eval_df.with_columns(
        pl.col("y").cast(pl.Int64), pl.col("y_hat_prob").cast(pl.Float64)
    )
    main_eval_df = EvalFrame(frame=eval_df, allow_extra_columns=True)
    eval_df = main_eval_df.frame

    plots: Sequence[SingleRunPlot] = [
        ROCByGroupPlot(group_auroc_model(runs=group_auroc_experiments)),
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

    structured_text_experiment = "ia_outpatient_all_features_training"
    structured_text_experiment_path = f"E:/shared_resources/forced_admissions_outpatient/eval_runs/{structured_text_experiment}_best_run_evaluated_on_test"
    structured_text_df = read_eval_df_from_disk(structured_text_experiment_path)

    feature_set_eval_dfs = {"Structured + text": structured_text_df}

    save_dir = Path(structured_text_experiment_path + "/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    figure = single_run_main(
        eval_df=structured_text_df,
        group_auroc_experiments=ExperimentWithNames(feature_set_eval_dfs),
        desired_positive_rate=0.02,
        outcome_label="Involuntary admissions",
        first_letter_index=0,
    )

    figure.savefig(save_dir / "fao_main_plot.png")
