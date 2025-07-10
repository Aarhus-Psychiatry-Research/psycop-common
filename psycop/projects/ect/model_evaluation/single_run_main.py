import logging
from pathlib import Path
from typing import TYPE_CHECKING

import patchworklib as pw
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    add_first_ect_time_after_prediction_time,
)
from psycop.projects.ect.model_evaluation.auroc_by.roc_by_multiple_runs_model import (
    ExperimentWithNames,
    group_auroc_model,
)
from psycop.projects.ect.model_evaluation.auroc_by.roc_by_multiple_runs_view import ROCByGroupPlot
from psycop.projects.ect.model_evaluation.confusion_matrix.model import confusion_matrix_model
from psycop.projects.ect.model_evaluation.confusion_matrix.view import ConfusionMatrixPlot
from psycop.projects.ect.model_evaluation.first_pos_pred_to_event.model import (
    first_positive_prediction_to_event_model,
)
from psycop.projects.ect.model_evaluation.first_pos_pred_to_event.view import (
    FirstPosPredToEventPlot,
)
from psycop.projects.ect.model_evaluation.sensitivity_by_time_to_event.model import (
    sensitivity_by_time_to_event_model,
)
from psycop.projects.ect.model_evaluation.sensitivity_by_time_to_event.view import (
    SensitivityByTTEPlot,
)
from psycop.projects.ect.model_evaluation.single_run_artifact import SingleRunPlot
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk
from psycop.projects.scz_bp.evaluation.configs import COLORS

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
    main_eval_df = EvalFrame(frame=eval_df, allow_extra_columns=True)
    eval_df = main_eval_df.frame

    eval_df_with_correct_time_to_outcome = add_first_ect_time_after_prediction_time(eval_df)

    plots: Sequence[SingleRunPlot] = [
        ROCByGroupPlot(group_auroc_model(runs=group_auroc_experiments)),
        ConfusionMatrixPlot(
            confusion_matrix_model(eval_df=eval_df, desired_positive_rate=desired_positive_rate),
            outcome_label=outcome_label,
        ),
        FirstPosPredToEventPlot(
            data=first_positive_prediction_to_event_model(
                eval_df=eval_df_with_correct_time_to_outcome,
                desired_positive_rate=desired_positive_rate,
            ),
            outcome_label=outcome_label,
        ),
        SensitivityByTTEPlot(
            outcome_label=outcome_label,
            data=sensitivity_by_time_to_event_model(eval_df=eval_df_with_correct_time_to_outcome),
            colors=COLORS,
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

    structured_text_experiment = "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter"
    structured_text_experiment_path = f"E:/shared_resources/ect/eval_runs/{structured_text_experiment}_best_run_evaluated_on_geographic_test"
    structured_text_df = read_eval_df_from_disk(structured_text_experiment_path)

    # read other dfs
    structured_only_experiment = "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter"
    structured_only_experiment_path = f"E:/shared_resources/ect/eval_runs/{structured_only_experiment}_best_run_evaluated_on_geographic_test"
    structured_only_df = read_eval_df_from_disk(structured_only_experiment_path)

    text_only_experiment = "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter"
    text_only_experiment_path = f"E:/shared_resources/ect/eval_runs/{text_only_experiment}_best_run_evaluated_on_geographic_test"
    text_only_df = read_eval_df_from_disk(text_only_experiment_path)

    feature_set_eval_dfs = {
        "Structured only": structured_only_df,
        "Text only": text_only_df,
        "Structured + text": structured_text_df,
    }

    feature_sets = ["structured_only", "text_only", "structured_text"]

    for feature_set_name in feature_sets:
        experiment = f"ECT-trunc-and-hp-{feature_set_name}-xgboost-no-lookbehind-filter"
        experiment_path = (
            f"E:/shared_resources/ect/eval_runs/{experiment}_best_run_evaluated_on_test"
        )
        experiment_df = read_eval_df_from_disk(experiment_path)

        save_dir = Path(experiment_path + "/figures")
        save_dir.mkdir(parents=True, exist_ok=True)

        figure = single_run_main(
            eval_df=experiment_df,
            group_auroc_experiments=ExperimentWithNames(feature_set_eval_dfs),
            desired_positive_rate=0.02,
            outcome_label="ECT",
            first_letter_index=0,
        )

        figure.savefig(save_dir / "ect_main_plot.png")
