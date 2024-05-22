import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
    cvd_pred_times,
)
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.age_model import auroc_by_age_model

if __name__ == "__main__":
    run = MlflowClientWrapper().get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
    pred_timestamps = cvd_pred_times()
    outcome_timestamps = cvd_outcome_timestamps()

    model = auroc_by_age_model(eval_df=run.eval_df(), birthdays=pl.from_pandas(birthdays()))
