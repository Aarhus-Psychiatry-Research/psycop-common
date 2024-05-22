import logging

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
    cvd_pred_times,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.day_of_week_model import (
    auroc_by_day_of_week_model,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.day_of_week_view import (
    AUROCByDayOfWeekPlot,
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
