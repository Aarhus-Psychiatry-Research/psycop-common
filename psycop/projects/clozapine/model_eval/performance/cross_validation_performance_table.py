from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper


def cross_validation_performance_table(
    models_to_train: pd.DataFrame, models_descriptions: str | None = None
) -> pd.DataFrame:
    internal_roc_aurocs = []
    cfs = []
    std_devs = []
    internal_auroc_intervals = []
    apparent_aurocs = []
    model_optimisms = []

    client = MlflowClientWrapper()

    for i in models_to_train.index:
        experiment_name = models_to_train["model_name"][i]

        # Step 1: Get best run by 'all_oof_BinaryAUROC'
        best_run = client.get_best_run_from_experiment(
            experiment_name=experiment_name,  # pyright: ignore
            metric="all_oof_BinaryAUROC",
        )

        run_name = best_run.name
        run_metrics = best_run.get_all_metrics().melt(id_vars="run_name", variable_name="metric")

        # Step 2: Get out-of-fold AUROC metrics
        oof_aucs = (
            run_metrics.filter(pl.col("metric").str.contains("out_of_fold"))
            .select("value")
            .to_series()
            .cast(pl.Float64)
            .to_list()
        )

        if not oof_aucs:
            print(f"⚠️ No OOF metrics found for run: {run_name}")
            continue

        within_fold_aucs = (
            run_metrics.filter(pl.col("metric").str.contains("within_fold"))
            .select("value")
            .to_series()
            .cast(pl.Float64)
            .to_list()
        )

        if not within_fold_aucs:
            print(f"⚠️ No within_fold metrics found for run: {run_name}")
            continue

        # Step 4: Statistical metrics
        internal_auroc = np.mean(oof_aucs)
        apparent_auroc = np.mean(within_fold_aucs)
        std_dev_auroc = np.std(oof_aucs)
        std_error_auroc = std_dev_auroc / np.sqrt(len(oof_aucs))
        dof = len(oof_aucs) - 1
        conf_interval = stats.t.interval(0.95, dof, loc=internal_auroc, scale=std_error_auroc)
        optimism = apparent_auroc - internal_auroc

        # Step 5: Append results
        internal_roc_aurocs.append(round(internal_auroc, 3))
        cfs.append(f"[{round(conf_interval[0], 3)}:{round(conf_interval[1], 3)}]")
        std_devs.append(round(std_dev_auroc, 3))
        internal_auroc_intervals.append(f"{round(min(oof_aucs), 3)}-{round(max(oof_aucs), 3)}")
        apparent_aurocs.append(round(apparent_auroc, 3))
        within_fold_aucs.append(
            f"{round(min(within_fold_aucs), 3)}-{round(max(within_fold_aucs), 3)}"
        )
        model_optimisms.append(round(optimism, 3))

        # Optional: training size can be logged as a param if available

    df_dict = {
        "Predictor set": models_to_train["pretty_model_name"],
        "Model type": models_to_train["pretty_model_type"],
        "Internal AUROC score": internal_roc_aurocs,
        "95 percent confidence interval": cfs,
        "Standard deviation": std_devs,
        "5-fold out-of-fold AUROC interval": internal_auroc_intervals,
        "Apparent AUROC score": apparent_aurocs,
        "Model optimism": model_optimisms,
    }
    # add this later "Number of training samples": n_train_rows, # "Number of outcomes in training data": n_outcomes,

    df = pd.DataFrame(df_dict)
    EVAL_ROOT = Path("E:/shared_resources/clozapine/eval")
    df.to_excel(EVAL_ROOT / f"{models_descriptions}cross_validation_table.xlsx", index=False)

    return df


if __name__ == "__main__":
    #### XGBOOST ####
    xg_df = pd.DataFrame(
        {
            "pretty_model_name": [
                "365d_lookahead_Structured + TF-IDF",
                "365d_lookahead_Structured",
                "365d_lookahead_TF-IDF 180 days",
                "365d_lookahead_Only unique_count_antipsychotics",
                "730d_lookahead_Structured + TF-IDF",
                "730d_lookahead_Structured",
                "730d_lookahead_only_TFIDF",
                "730d_lookahead_Only unique_count_antipsychotics",
            ],
            "model_name": [
                "clozapine hparam, structured_text_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, only_structured_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, only_tfidf_365d_lookahead, xgboost, 1 year lookbehind filter,2025_random_split",
                "clozapine hparam, unique_antipsychotics_365d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, structured_text_730d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, only_structured_730d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, only_tfidf_730d_lookahead, xgboost, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, unique_antipsychotics_730d_lookahead, xgboost, 1 year lookbehind filter,2025_random_split",
            ],
            # change this for either XGboost or Logistic regression
            "pretty_model_type": [
                "XGBoost",
                "XGBoost",
                "XGBoost",
                "XGBoost",
                "XGBoost",
                "XGBoost",
                "XGBoost",
                "XGBoost",
            ],
        }
    )

    cross_validation_performance_table(xg_df, "primary_models_xgboost_1y_lookbehind_filter_")

    log_reg_df = pd.DataFrame(
        {
            "pretty_model_name": [
                "365d_lookahead_Structured + TF-IDF",
                "365d_lookahead_Structured",
                "365d_lookahead_TF-IDF 180 days",
                "365d_lookahead_Only unique_count_antipsychotics",
                "730d_lookahead_Structured + TF-IDF",
                "730d_lookahead_Structured",
                "730d_lookahead_only_TFIDF",
                "730d_lookahead_Only unique_count_antipsychotics",
            ],
            "model_name": [
                "clozapine hparam, structured_text_365d_lookahead, log_reg, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, only_structured_365d_lookahead, log_reg, 1 year lookbehind filter,2025_random_split",
                "clozapine hparam, only_tfidf_365d_lookahead, log_reg, 1 year lookbehind filter,2025_random_split",
                "clozapine hparam, unique_antipsychotics_365d_lookahead, log_reg, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, structured_text_730d_lookahead, log_reg, 1 year lookbehind filter,2025_random_split",
                "clozapine hparam, only_structured_730d_lookahead, log_reg, 1 year lookbehind filter, 2025_random_split",
                "clozapine hparam, only_tfidf_730d_lookahead, log_reg, 1 year lookbehind filter,2025_random_split",
                "clozapine hparam, unique_antipsychotics_730d_lookahead, log_reg, 1 year lookbehind filter, 2025_random_split",
            ],
            # change this for either XGboost or Logistic regression
            "pretty_model_type": [
                "Log_reg",
                "Log_reg",
                "Log_reg",
                "Log_reg",
                "Log_reg",
                "Log_reg",
                "Log_reg",
                "Log_reg",
            ],
        }
    )

    cross_validation_performance_table(log_reg_df, "primary_models_log_reg_1y_lookbehind_filter_")
