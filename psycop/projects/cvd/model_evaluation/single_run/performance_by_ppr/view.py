import pandas as pd
import polars as pl

from psycop.projects.cvd.model_evaluation.single_run.performance_by_ppr.model import (
    PerformanceByPPRModel,
)


def _format_with_thousand_separator(num: int) -> str:
    return f"{num:,.0f}"


def _format_prop_as_percent(num: float) -> str:
    output = f"{num:.1%}"

    return output


def performance_by_ppr_view(model: PerformanceByPPRModel) -> pl.DataFrame:
    df = model

    output_df = df.drop(
        [
            "total_warning_days",
            "warning_days_per_false_positive",
            "negative_rate",
            "mean_warning_days",
            "median_warning_days",
            "prop with ≥1 true positive",
        ]
    )

    renamed_df = output_df.rename(
        {
            "positive_rate": "Predicted positive rate",
            "true_prevalence": "True prevalence",
            "sensitivity": "Sens",
            "specificity": "Spec",
            "accuracy": "Acc",
            "true_positives": "TP",
            "true_negatives": "TN",
            "false_positives": "FP",
            "false_negatives": "FN",
            "prop_of_all_events_captured": "% of all T2D captured",
            "f1": "F1",
        }
    )

    # Handle proportion columns
    prop_cols = [col for col, dtype in renamed_df.schema.items() if dtype == pl.Float64]
    for c in prop_cols:
        renamed_df = renamed_df.with_columns([pl.col(c).apply(lambda x: f"{x:.1%}").alias(c)])

    # Handle count columns
    count_cols = [col for col, dtype in renamed_df.schema.items() if dtype == pl.Int64]
    for col in count_cols:
        renamed_df = renamed_df.with_columns(pl.col(col).apply(lambda x: f"{x:,}").alias(col))

    renamed_df = renamed_df.with_columns(
        model.with_columns(
            (pl.col("median_warning_days") / 365.25)
            .round(1)
            .alias("Median years from first positive to CVD")
        )
    )

    return renamed_df
