import logging
from collections.abc import Sequence

import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.common.model_evaluation.markdown.md_objects import MarkdownFigure
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
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
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot

log = logging.getLogger(__name__)

if __name__ == "__main__":
    eval_frame = (
        MlflowClientWrapper()
        .get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
        .eval_frame()
    )
    eval_df = eval_frame.frame

    plots: Sequence[SingleRunPlot] = [
        AUROCBySex(auroc_by_sex_model(eval_df=eval_df, sex_df=pl.from_pandas(sex_female()))),
        AUROCByAge(
            auroc_by_age_model(
                eval_df=eval_df,
                birthdays=pl.from_pandas(birthdays()),
                bins=[18, *range(20, 80, 10)],
            )
        ),
        AUROCByTimeFromFirstVisitPlot(
            auroc_by_time_from_first_visit_model(
                eval_frame=eval_frame, all_visits_df=pl.from_pandas(physical_visits_to_psychiatry())
            )
        ),
        AUROCByMonthOfYearPlot(auroc_by_month_of_year_model(eval_frame=eval_frame)),
        AUROCByDayOfWeekPlot(auroc_by_day_of_week_model(eval_frame=eval_frame)),
    ]

    ggplots: list[pn.ggplot] = []
    for plot in plots:
        log.info(f"Starting processing of {plot.__class__.__name__}")
        ggplots.append(plot())

    figure = create_patchwork_grid(plots=ggplots, single_plot_dimensions=(5, 3), n_in_row=2)
    figure.savefig("test.png")
