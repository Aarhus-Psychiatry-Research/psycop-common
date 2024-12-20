import logging
from typing import TYPE_CHECKING

import patchworklib as pw
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame, MlflowClientWrapper
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.age_model import auroc_by_age_model
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.age_view import AUROCByAge
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.month_of_year_model import (
    auroc_by_month_of_year_model,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.month_of_year_view import (
    AUROCByMonthOfYearPlot,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.sex_model import auroc_by_sex_model
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.sex_view import AUROCBySex
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.time_from_first_visit_model import (
    auroc_by_time_from_first_visit_model,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.time_from_first_visit_view import (
    AUROCByTimeFromFirstVisitPlot,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.single_run_artifact import SingleRunPlot

if TYPE_CHECKING:
    from collections.abc import Sequence

    import plotnine as pn

log = logging.getLogger(__name__)


def single_run_robustness(
    eval_frame: EvalFrame,
    sex_df: pl.DataFrame,
    birthdays: pl.DataFrame,
    all_visits_df: pl.DataFrame,
) -> pw.Bricks:
    eval_df = eval_frame.frame

    plots: Sequence[SingleRunPlot] = [
        AUROCBySex(auroc_by_sex_model(eval_df=eval_df, sex_df=sex_df)),
        AUROCByAge(
            auroc_by_age_model(eval_df=eval_df, birthdays=birthdays, bins=[18, *range(20, 80, 10)])
        ),
        AUROCByTimeFromFirstVisitPlot(
            auroc_by_time_from_first_visit_model(eval_frame=eval_frame, all_visits_df=all_visits_df)
        ),
        AUROCByMonthOfYearPlot(auroc_by_month_of_year_model(eval_frame=eval_frame)),
    ]

    ggplots: list[pn.ggplot] = []
    for plot in plots:
        log.info(f"Starting processing of {plot.__class__.__name__}")
        ggplots.append(plot())

    figure = create_patchwork_grid(plots=ggplots, single_plot_dimensions=(5, 4.5), n_in_row=2)
    return figure


if __name__ == "__main__":
    import coloredlogs

    run_name = "2018-01-01_2018-01-01_2018-12-31"

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    eval_frame = (
        MlflowClientWrapper()
        .get_run(experiment_name="T2D-extended, temporal validation",
            run_name=run_name,)
        .eval_frame()
    )

    figure = single_run_robustness(
        eval_frame=eval_frame,
        birthdays=pl.from_pandas(birthdays()),
        sex_df=pl.from_pandas(sex_female()),
        all_visits_df=pl.from_pandas(physical_visits_to_psychiatry()),
    )

    figure.savefig(f"t2d-extended_{run_name}_robustness.png")
