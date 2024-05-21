from itertools import product

import pandas as pd
import plotnine as pn
import polars as pl
from sklearn.metrics import roc_auc_score

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper


def create_permutations(
    model_names: list[str], validation_outcome_col_names: list[str]
) -> list[tuple[str, str]]:
    return list(product(model_names, validation_outcome_col_names))


def auroc_by_outcome(
    model_names: list[str],
    validation_outcomes: list[pl.DataFrame],
    y_hat_col_name: str = "y_hat_prob",
    best_run_metric: str = "all_oof_BinaryAUROC",
) -> pd.DataFrame:
    validation_outcome_col_names = [
        validation_outcome.columns[1] for validation_outcome in validation_outcomes
    ]
    performance_dfs = []
    models_by_outcomes = create_permutations(model_names, validation_outcome_col_names)  # type: ignore

    for model_name, validation_outcome_col_name in models_by_outcomes:
        eval_df = (
            MlflowClientWrapper()
            .get_best_run_from_experiment(experiment_name=model_name, metric=best_run_metric)
            .eval_df()
        )
        validation_outcome = next(
            validation_outcome
            for validation_outcome in validation_outcomes
            if validation_outcome.columns[1] == validation_outcome_col_name
        )

        joined_df = eval_df.join(
            validation_outcome, on="pred_time_uuid", how="left", validate="1:1"
        ).filter(pl.col(validation_outcome_col_name).is_not_null())
        performance_dfs.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "validation_outcome": validation_outcome_col_name,
                    "estimate": [
                        round(
                            roc_auc_score(  # type: ignore
                                y_true=joined_df[validation_outcome_col_name],
                                y_score=joined_df[y_hat_col_name],
                            ),
                            2,
                        )
                    ],
                }
            )
        )

    performance_df = pd.concat(performance_dfs)
    return performance_df


def plot_auroc_by_outcome(
    df: pd.DataFrame,
    x_axis_label: str = "Validation outcome",
    y_axis_label: str = "Training outcome",
    plot_title: str = "AUROC by Outcome",
) -> pn.ggplot:
    p = (
        pn.ggplot(df, pn.aes(x="validation_outcome", y="model_name", fill="estimate"))
        + pn.geom_tile(pn.aes(width=0.95, height=0.95), fill="gainsboro")
        + pn.geom_text(pn.aes(label="estimate"), size=18, color="black")
        + pn.theme(
            axis_line=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            axis_text=pn.element_text(size=10, color="black"),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            panel_background=pn.element_blank(),
            legend_position="none",
            plot_title=pn.element_text(size=15, color="black", ha="center"),
        )
        + pn.labs(x=x_axis_label, y=y_axis_label, title=plot_title)
    )

    return p
