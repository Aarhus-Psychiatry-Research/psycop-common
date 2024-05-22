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

if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    run = MlflowClientWrapper().get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
    pred_timestamps = cvd_pred_times()
    outcome_timestamps = cvd_outcome_timestamps()

    model = auroc_by_time_from_first_visit_model(
        eval_df=run.eval_df(), all_visits_df=pl.from_pandas(physical_visits_to_psychiatry())
    )
    plot = AUROCByTimeFromFirstVisitPlot(data=model)()
    plot.save("test.png")
    pass
