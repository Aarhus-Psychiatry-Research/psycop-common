from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
    cvd_pred_times,
)
from psycop.projects.cvd.model_evaluation.single_run.performance_by_ppr.model import (
    performance_by_ppr_model,
)
from psycop.projects.cvd.model_evaluation.single_run.performance_by_ppr.view import (
    performance_by_ppr_view,
)

if __name__ == "__main__":
    run = MlflowClientWrapper().get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
    eval_df = run.eval_df()
    prediction_timestamps = cvd_pred_times()
    outcome_timestamps = cvd_outcome_timestamps()

    model = performance_by_ppr_model(
        eval_df=eval_df,
        pred_timestamps=prediction_timestamps,
        outcome_timestamps=outcome_timestamps,
        positive_rates=[0.05, 0.1],
    )

    view = performance_by_ppr_view(model=model)

