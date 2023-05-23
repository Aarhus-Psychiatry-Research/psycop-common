from pathlib import Path

import pandas as pd
from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, TABLES_PATH
from psycop.projects.t2d.utils.best_runs import ModelRun


def format_with_thousand_separator(num: int) -> str:
    return f"{num:,.0f}"


def format_prop_as_percent(num: float) -> str:
    output = f"{num:.1%}"

    return output


def clean_up_performance_by_ppr(table: pd.DataFrame) -> pd.DataFrame:
    df = table

    output_df = df.drop(
        [
            "total_warning_days",
            "warning_days_per_false_positive",
            "negative_rate",
            "mean_warning_days",
            "prop with ≥1 true positive",
        ],
        axis=1,
    )

    renamed_df = output_df.rename(
        {
            "positive_rate": "Positive rate",
            "true_prevalence": "True prevalence",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "accuracy": "Accuracy",
            "true_positives": "True positives",
            "true_negatives": "True negatives",
            "false_positives": "False positives",
            "false_negatives": "False negatives",
            "prop of all events captured": "% of all events captured",
        },
        axis=1,
    )

    # Handle proportion columns
    prop_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "float64"]
    for c in prop_cols:
        renamed_df[c] = renamed_df[c].apply(format_prop_as_percent)

    # Handle count columns
    count_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "int64"]
    for col in count_cols:
        renamed_df[col] = renamed_df[col].apply(format_with_thousand_separator)

    renamed_df["Mean years from first positive to T2D"] = round(
        df["mean_warning_days"] / 365,
        1,
    )

    return renamed_df


def output_performance_by_ppr(run: ModelRun) -> Path:
    eval_dataset = run.get_eval_dataset()

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset,
        positive_rates=[0.05, 0.04, 0.03, 0.02, 0.01],
    )

    df = clean_up_performance_by_ppr(df)

    table_path = TABLES_PATH / "performance_by_ppr.xlsx"
    TABLES_PATH.mkdir(exist_ok=True, parents=True)
    df.to_excel(table_path)

    return table_path


if __name__ == "__main__":
    output_performance_by_ppr(run=EVAL_RUN)
