import pandas as pd
import polars as pl
from itertools import product
import plotnine as pn
from sklearn.metrics import roc_auc_score

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper


def create_permutations(model_names: list[str], validation_outcome_col_names: list[str]) -> list[tuple[str, str]]:
    return list(product(model_names, validation_outcome_col_names))


def auroc_by_outcome(model_names: list[str], validation_outcomes: list[pl.DataFrame], y_hat_col_name: str = "y_hat_prob", best_run_metric: str = "all_oof_BinaryAUROC") -> pd.DataFrame:
    validation_outcome_col_names = [validation_outcome.columns[1] for validation_outcome in validation_outcomes]
    performance_df = pd.DataFrame(columns=validation_outcome_col_names)
    models_by_outcomes = create_permutations(model_names, validation_outcome_col_names) # type: ignore

    for model_name, validation_outcome_col_name in models_by_outcomes:
        eval_df = MlflowClientWrapper().get_best_run_from_experiment(experiment_name=model_name, metric=best_run_metric).eval_df()
        validation_outcome = [validation_outcome for validation_outcome in validation_outcomes if validation_outcome.columns[1] == validation_outcome_col_name][0]
    
        joined_df = eval_df.join(validation_outcome, on="pred_time_uuid", how="left", validate="1:1").filter(pl.col(validation_outcome_col_name).is_not_null())
        performance_df.loc[model_name, validation_outcome_col_name] = roc_auc_score( # type: ignore
            y_true=joined_df[validation_outcome_col_name], y_score=joined_df[y_hat_col_name]
        )

    return performance_df


def plot_auroc_by_outcome() -> pn.ggplot:
    return pn.ggplot()