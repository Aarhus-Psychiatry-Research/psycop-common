from typing import Callable
import numpy as np
import pandas as pd

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.binary.bootstrap_estimates import bootstrap_estimates
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk


def get_training_performance_cis(
    base_path: str,
    experiments: list[str],
    metric: Callable[[pd.Series, pd.Series], float],
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
        positive_rate (float, optional): Desired positive rate for thresholding. Defaults to 0.03
        n_bootstrap_samples (int, optional): Number of bootstrap samples. Defaults to 1000.
        ci (int, optional): Confidence interval width (in percentage). Defaults to 95.

    Returns:
        pd.DataFrame: DataFrame containing confidence intervals for each experiment.
    """
    ci_results = []
    experiment_names = []

    for experiment in experiments:
        path = f"{base_path}/{experiment}/eval_df.parquet"

        df = read_eval_df_from_disk(path).to_pandas()

        y_true = df["y"]
        y_hat_probs = df["y_hat_prob"]
        y_pred = get_predictions_for_positive_rate(
            desired_positive_rate=positive_rate,
            y_hat_probs=df["y_hat_prob"],  # type: ignore
        )[0]

        boot_ci = bootstrap_estimates(
            y_true=y_true,  # type: ignore
            y_pred=y_hat_probs if metric.__name__ == "roc_auc" else y_pred,  # type: ignore
            metric=metric,
            ci_width=ci / 100,
            n_resamples=n_bootstrap_samples,
            method="basic",
            random_state=42,
            stratified=True,
        )

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


if __name__ == "__main__":
    ci_df = get_training_performance_cis(
        base_path="E:/shared_resources/ect/eval_runs",
        experiments=[
            "ECT-trunc-and-hp-structured_only-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test",
            "ECT-trunc-and-hp-text_only-xgboost-with-lookbehind-filter_best_run_evaluated_on_test",
        ],
        metric=lambda y_true, y_pred: np.mean(y_true == y_pred),  # type: ignore
        positive_rate=0.03,
        n_bootstrap_samples=1000,
        ci=95,
        file_name="accuracy_ci",
    )
    print(ci_df)
