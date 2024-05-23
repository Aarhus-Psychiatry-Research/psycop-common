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


def performance_by_ppr_view(table: PerformanceByPPRModel) -> pl.DataFrame:
    df = table

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
            "prop of all events captured": "% of all T2D captured",
            "f1": "F1",
            "mcc": "MCC",
        }
    )

    # Handle proportion columns
    prop_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "float64"]
    for c in prop_cols:
        renamed_df[c] = renamed_df[c].apply(_format_prop_as_percent)

    # Handle count columns
    count_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "int64"]
    for col in count_cols:
        renamed_df[col] = renamed_df[col].apply(_format_with_thousand_separator)

    renamed_df["Median years from first positive to T2D"] = renamed_df.with_columns(
        (pl.col("median_warning_days") / 365.25)
        .round(1)
        .alias("Median years from first positive to T2D")
    )

    return renamed_df
