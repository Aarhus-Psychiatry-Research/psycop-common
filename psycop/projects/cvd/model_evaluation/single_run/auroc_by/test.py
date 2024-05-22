import logging

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
    cvd_pred_times,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.age_model import auroc_by_age_model
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.age_view import AUROCByAge
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.day_of_week_model import (
    auroc_by_day_of_week_model,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.day_of_week_view import (
    AUROCByDayOfWeekPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.month_of_year_model import (
    auroc_by_month_of_year,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.month_of_year_view import (
    AUROCByMonthOfYearPlot,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.sex_model import (
    AurocBySexDF,
    auroc_by_sex_model,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.sex_view import AUROCBySex
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.time_from_first_visit_model import (
    auroc_by_time_from_first_visit_model,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.time_from_first_visit_view import (
    AUROCByTimeFromFirstVisitPlot,
)
from psycop.projects.restraint.model_evaluation.application.pipelines.robustness.robustness_by_cyclic_time import (
    auroc_by_day_of_week,
)

if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    logging.info("Testing")
    run = MlflowClientWrapper().get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
    pred_timestamps = cvd_pred_times()
    outcome_timestamps = cvd_outcome_timestamps()

    model = auroc_by_day_of_week_model(eval_df=run.eval_df())
    plot = AUROCByDayOfWeekPlot(data=model)()
    plot.save("test.png")
    pass
