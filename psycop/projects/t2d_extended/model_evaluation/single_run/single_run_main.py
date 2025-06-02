import logging
from typing import TYPE_CHECKING

import patchworklib as pw

from psycop.common.cohort_definition import OutcomeTimestampFrame
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame, MlflowClientWrapper
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.t2d_extended.feature_generation.cohort_definition.t2d_cohort_definer import (
    t2d_outcome_timestamps,
    t2d_pred_times,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc.model import auroc_model
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc.view import AUROCPlot
from psycop.projects.t2d_extended.model_evaluation.single_run.confusion_matrix.model import (
    confusion_matrix_model,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.confusion_matrix.view import (
    ConfusionMatrixPlot,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.first_pos_pred_to_event.model import (
    first_positive_prediction_to_event_model,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.first_pos_pred_to_event.view import (
    FirstPosPredToEventPlot,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    sensitivity_by_time_to_event_model,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.sensitivity_by_time_to_event.view import (
    SensitivityByTTEPlot,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.scz_bp.evaluation.configs import COLORS

if TYPE_CHECKING:
    from collections.abc import Sequence

    import plotnine as pn

log = logging.getLogger(__name__)


def single_run_main(
    eval_frame: EvalFrame,
    desired_positive_rate: float,
    outcome_label: str,
    outcome_timestamps: OutcomeTimestampFrame,
    first_letter_index: int,
) -> pw.Bricks:
    eval_df = eval_frame.frame

    plots: Sequence[SingleRunPlot] = [
        AUROCPlot(auroc_model(eval_df=eval_df)),
        ConfusionMatrixPlot(
            confusion_matrix_model(eval_df=eval_df, desired_positive_rate=desired_positive_rate),
            outcome_label=outcome_label,
        ),
        SensitivityByTTEPlot(
            outcome_label=outcome_label,
            data=sensitivity_by_time_to_event_model(
                eval_df=eval_df, outcome_timestamps=outcome_timestamps, pprs=[desired_positive_rate]
            ),
            colors=COLORS,
        ),
        FirstPosPredToEventPlot(
            data=first_positive_prediction_to_event_model(
                eval_df=eval_df, outcome_timestamps=outcome_timestamps
            ),
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

    run_name = "2018-01-01_2022-01-01_2022-12-31"

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    eval_frame = (
        MlflowClientWrapper()
        .get_run(
            experiment_name="T2D-extended, temporal validation",
            run_name=run_name,
        )
        .eval_frame()
    )
    pred_timestamps = t2d_pred_times()
    outcome_timestamps = t2d_outcome_timestamps()

    figure = single_run_main(
        eval_frame=eval_frame,
        desired_positive_rate=0.05,
        outcome_label="t2d",
        outcome_timestamps=outcome_timestamps,
        first_letter_index=0,
    )

    figure.savefig(f"t2d-extended_{run_name}_main.png")
