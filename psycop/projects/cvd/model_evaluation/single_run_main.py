import logging
from typing import TYPE_CHECKING

import patchworklib as pw
import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame, PredictionTimeFrame
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    EvalFrame,
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.common.model_evaluation.markdown.md_objects import MarkdownFigure
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
    cvd_pred_times,
)
from psycop.projects.cvd.model_evaluation.single_run import sensitivity_by_time_to_event
from psycop.projects.cvd.model_evaluation.single_run.auroc.model import auroc_model
from psycop.projects.cvd.model_evaluation.single_run.auroc.view import AUROCPlot
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.age_model import auroc_by_age_model
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.age_view import AUROCByAge
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.day_of_week_model import (
    auroc_by_day_of_week_model,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.day_of_week_view import (
    AUROCByDayOfWeekPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.month_of_year_model import (
    auroc_by_month_of_year_model,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.month_of_year_view import (
    AUROCByMonthOfYearPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.sex_model import auroc_by_sex_model
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.sex_view import AUROCBySex
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.time_from_first_visit_model import (
    auroc_by_time_from_first_visit_model,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.time_from_first_visit_view import (
    AUROCByTimeFromFirstVisitPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.confusion_matrix.model import (
    confusion_matrix_model,
)
from psycop.projects.cvd.model_evaluation.single_run.confusion_matrix.view import (
    ConfusionMatrixPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.first_pos_pred_to_event.model import (
    first_positive_prediction_to_event_model,
)
from psycop.projects.cvd.model_evaluation.single_run.first_pos_pred_to_event.view import (
    FirstPosPredToEventPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    sensitivity_by_time_to_event_model,
)
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.view import (
    SensitivityByTTEPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot

if TYPE_CHECKING:
    from collections.abc import Sequence

    import plotnine as pn

log = logging.getLogger(__name__)


def single_run_main(
    eval_frame: EvalFrame,
    desired_positive_rate: float,
    outcome_label: str,
    outcome_timestamps: OutcomeTimestampFrame,
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

    figure = create_patchwork_grid(plots=ggplots, single_plot_dimensions=(5, 4.5), n_in_row=2)
    return figure


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    eval_frame = (
        MlflowClientWrapper()
        .get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
        .eval_frame()
    )
    pred_timestamps = cvd_pred_times()
    outcome_timestamps = cvd_outcome_timestamps()

    figure = single_run_main(
        eval_frame=eval_frame,
        desired_positive_rate=0.05,
        outcome_label="CVD",
        outcome_timestamps=outcome_timestamps,
    )

    figure.savefig("test_cvd_main.png")
