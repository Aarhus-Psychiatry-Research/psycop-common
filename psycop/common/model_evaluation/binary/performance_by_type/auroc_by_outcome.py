from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import pandas as pd
import plotnine as pn
import polars as pl
from sklearn.metrics import roc_auc_score

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper


@dataclass
class EvaluationFrame:
    """Dataclass used for evaluating AUROC by different combinations of models
    and outcomes. Df should contain 2 columns: 1 for the prediction time uuid
    and 1 with the outcome (0/1)"""

    df: pl.DataFrame
    outcome_col_name: str

    def __post_init__(self):
        cols = self.df.columns
        if len(cols) > 2:
            raise ValueError(
                f"Df should only contain 2 columns: a prediction time uuid, and an outcome column. Current columns: {self.df.columns}"
            )


def create_permutations(
    model_names: list[str], validation_outcome_col_names: list[str]
) -> list[tuple[str, str]]:
    return list(product(model_names, validation_outcome_col_names))


def auroc_by_outcome(
    model_names: list[str],
    validation_outcomes: Sequence[EvaluationFrame],
    y_hat_col_name: str = "y_hat_prob",
    prediction_time_uuid: str = "pred_time_uuid",
    best_run_metric: str = "all_oof_BinaryAUROC",
) -> pd.DataFrame:
    validation_outcome_col_names = [
        validation_outcome.outcome_col_name for validation_outcome in validation_outcomes
    ]

    performance_dfs = []
    models_by_outcomes = create_permutations(model_names, validation_outcome_col_names)  # type: ignore

    for model_name, validation_outcome_col_name in models_by_outcomes:
        eval_df = (
            MlflowClientWrapper()
            .get_best_run_from_experiment(experiment_name=model_name, metric=best_run_metric)
            .eval_frame()
            .frame
        )
        validation_outcome = next(
            validation_outcome
            for validation_outcome in validation_outcomes
            if validation_outcome.outcome_col_name == validation_outcome_col_name
        )

        joined_df = eval_df.join(
            validation_outcome.df, on=prediction_time_uuid, how="left", validate="1:1"
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
