from typing import Callable

import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.binary.bootstrap_estimates import bootstrap_estimates
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    add_first_ect_time_after_prediction_time,
)
from psycop.projects.ect.model_evaluation.performance_by_ppr.days_from_first_positive_to_event import (
    _get_time_from_first_positive_to_diagnosis_df,
)
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk


def get_training_performance_cis(
    base_path: str,
    experiments: list[str],
    metric: Callable[[pd.Series, pd.Series], float],  # type: ignore
    from_disk: bool = True,
    positive_rate: float = 0.03,
    n_bootstrap_samples: int = 1000,
    ci: int = 95,
    file_name: str = "confidence_intervals",
) -> pd.DataFrame:
    """
    Function for calculating confidence intervals for training performance
    across multiple experiments.

    Args:
        base_path (str): Base path where evaluation dataframes are stored.
        experiments (list[str]): List of experiment names to evaluate.
        metric (Callable[[pd.Series, pd.Series], float]): Metric function to evaluate.
        from_disk (bool): Whether the eval df for the experiment is logged on disk or needs to be generated from a config on MLflow. Defaults to True.
        positive_rate (float, optional): Desired positive rate for thresholding. Defaults to 0.03
        n_bootstrap_samples (int, optional): Number of bootstrap samples. Defaults to 1000.
        ci (int, optional): Confidence interval width (in percentage). Defaults to 95.
        file_name (str): Name of the output file

    Returns:
        pd.DataFrame: DataFrame containing confidence intervals for each experiment.
    """
    ci_results = []
    experiment_names = []

    for experiment in experiments:
        path = f"{base_path}/{experiment}"

        if not from_disk:
            best_run_cfg = (
                MlflowClientWrapper()
                .get_best_run_from_experiment(
                    experiment_name=experiment, metric="all_oof_BinaryAUROC"
                )
                .get_config()
            )

            _ = train_baseline_model_from_cfg(best_run_cfg)  # type: ignore

        df = read_eval_df_from_disk(path).to_pandas()

        y_true = df["y"]
        y_hat_probs = df["y_hat_prob"]
        y_pred = get_predictions_for_positive_rate(
            desired_positive_rate=positive_rate,
            y_hat_probs=df["y_hat_prob"],  # type: ignore
        )[0]

        boot_ci = bootstrap_estimates(
            metric,
            n_bootstraps=n_bootstrap_samples,
            ci_width=ci / 100,
            input_1=y_true,
            input_2=y_hat_probs if metric.__name__ == "roc_auc_score" else y_pred,  # type: ignore,
        )["ci"]

        ci_results.append(boot_ci)
        experiment_names.append(experiment)

    ci_df = pd.DataFrame(
        {
            "experiment": experiment_names,
            "ci_lower": [ci[0] for ci in ci_results],
            "ci_upper": [ci[1] for ci in ci_results],
        }
    )

    ci_df.to_csv(f"E:/shared_resources/ect/eval_runs/tables/{file_name}.csv")

    return ci_df


def get_median_time_from_first_positive_to_event(
    base_path: str,
    experiments: list[str],
    from_disk: bool = True,
    positive_rate: float = 0.02,
    file_name: str = "median_time_from_first_positive_to_event",
) -> pd.DataFrame:
    """
    Function to calculate the median time from first positive to event occurrence.

    Args:
        base_path (str): Base path where evaluation dataframes are stored.
        experiments (list[str]): List of experiment names to evaluate.
        from_disk (bool): Whether the eval df for the experiment is logged on disk or needs to be generated from a config on MLflow. Defaults to True.
        positive_rate (float, optional): Desired positive rate for thresholding. Defaults to 0.03
        file_name (str): Name of the output file

    Returns:
        float: Median time from first positive to event occurrence.
    """
    medians = []
    quantiles = []
    experiment_names = []

    for experiment in experiments:
        path = f"{base_path}/{experiment}"

        if not from_disk:
            best_run_cfg = (
                MlflowClientWrapper()
                .get_best_run_from_experiment(
                    experiment_name=experiment, metric="all_oof_BinaryAUROC"
                )
                .get_config()
            )

            _ = train_baseline_model_from_cfg(best_run_cfg)  # type: ignore

        eval_df = read_eval_df_from_disk(path)
        eval_df = add_first_ect_time_after_prediction_time(eval_df).to_pandas()

        df = pd.DataFrame(
            {
                "id": eval_df["dw_ek_borger"],
                "pred": get_predictions_for_positive_rate(
                    desired_positive_rate=positive_rate, y_hat_probs=eval_df["y_hat_prob"]
                )[0],
                "y": eval_df["y"],
                "pred_timestamps": eval_df["timestamp"],
                "outcome_timestamps": eval_df["timestamp_outcome"],
            }
        )

        df = _get_time_from_first_positive_to_diagnosis_df(input_df=df)  # type: ignore
        median = df["days_from_pred_to_event"].agg("median")
        quantile = df["days_from_pred_to_event"].quantile([0.25, 0.75])

        experiment_names.append(experiment)
        medians.append(median)
        quantiles.append(quantile)

    median_df = pd.DataFrame(
        {
            "experiment": experiment_names,
            "median_days_from_first_positive_to_event": medians,
            "25th_percentile_days_from_first_positive_to_event": [q[0.25] for q in quantiles],  # type: ignore
            "75th_percentile_days_from_first_positive_to_event": [q[0.75] for q in quantiles],  # type: ignore
        }
    )

    median_df.to_csv(f"E:/shared_resources/ect/eval_runs/tables/{file_name}.csv")

    return median_df


if __name__ == "__main__":
    median_df = get_median_time_from_first_positive_to_event(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_geographic_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_geographic_test",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_geographic_test",
        ],
        positive_rate=0.02,
        file_name="eval_on_geographic_test_median_time_from_first_positive_to_event",
    )
    print(median_df)

    ci_df = get_training_performance_cis(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_geographic_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_geographic_test",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_geographic_test",
        ],
        metric=roc_auc_score,  # type: ignore
        positive_rate=0.02,
        n_bootstrap_samples=1000,
        ci=95,
        file_name="eval_on_geographic_test_auroc_cis",
    )
    print(ci_df)

    median_df = get_median_time_from_first_positive_to_event(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
        ],
        positive_rate=0.02,
        file_name="eval_on_test_median_time_from_first_positive_to_event",
    )
    print(median_df)

    ci_df = get_training_performance_cis(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
        ],
        metric=recall_score,  # type: ignore
        positive_rate=0.02,
        n_bootstrap_samples=1000,
        ci=95,
        file_name="eval_on_test_sensitivity_cis",
    )
    print(ci_df)

    ci_df = get_training_performance_cis(
        base_path="E:/shared_resources/ect/training",
        experiments=[
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter",
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter",
        ],
        metric=roc_auc_score,  # type: ignore
        positive_rate=0.02,
        n_bootstrap_samples=1000,
        ci=95,
        file_name="training_cv_auroc_cis",
    )
    print(ci_df)

    ci_df = get_training_performance_cis(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
        ],
        metric=roc_auc_score,  # type: ignore
        positive_rate=0.02,
        n_bootstrap_samples=1000,
        ci=95,
        file_name="eval_on_test_auroc_cis",
    )
    print(ci_df)

    ci_df = get_training_performance_cis(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
        ],
        metric=recall_score,  # type: ignore
        positive_rate=0.02,
        n_bootstrap_samples=1000,
        ci=95,
        file_name="eval_on_test_sensitivity_cis",
    )
    print(ci_df)

    ci_df = get_training_performance_cis(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-text_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
        ],
        metric=recall_score,  # type: ignore
        positive_rate=0.02,
        n_bootstrap_samples=1000,
        ci=95,
        file_name="eval_on_test_sensitivity_cis",
    )
    print(ci_df)
