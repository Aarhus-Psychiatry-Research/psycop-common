from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from wasabi import Printer

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.ect.model_evaluation.utils import (
    expand_eval_df_with_extra_cols,
    read_eval_df_from_disk,
)

msg = Printer(timestamp=True)


def _df_to_eval_dataset(df: pd.DataFrame) -> EvalDataset:
    """Convert dataframe to EvalDataset."""
    return EvalDataset(
        ids=df["dw_ek_borger"],
        pred_time_uuids=df["pred_time_uuid"],
        y=df["y"],
        y_hat_probs=df["y_hat_prob"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_outcome"],
        age=df["age"],
    )


def _get_num_of_unique_outcome_events(eval_dataset: EvalDataset) -> int:
    df = pd.DataFrame(
        {
            "id": eval_dataset.ids,
            "y": eval_dataset.y,
            "pred_timestamps": eval_dataset.pred_timestamps,
            "outcome_timestamps": eval_dataset.outcome_timestamps,
        }
    )

    num_unique = df[df["y"] == 1][["id", "outcome_timestamps"]].drop_duplicates().shape[0]

    return num_unique


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
            "median_warning_days",
            "prop with ≥1 true positive",
            "prop of all events captured",
            "total_warning_days_for_true_positives",
        ],
        axis=1,
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
            "f1": "F1",
            "mcc": "MCC",
            "mean_warning_days_for_true_positives": "Mean warning days for TPs",
            "median_warning_days_for_true_positives": "Median warning days for TPs",
        },
        axis=1,
    )

    # Handle proportion columns
    prop_cols = [
        c
        for c in renamed_df.columns
        if renamed_df[c].dtype == "float64"
        and c not in {"Mean warning days for TPs", "Median warning days for TPs"}
    ]

    for c in prop_cols:
        renamed_df[c] = renamed_df[c].apply(format_prop_as_percent)

    # Handle count columns
    count_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "int64"]
    for col in count_cols:
        renamed_df[col] = renamed_df[col].apply(format_with_thousand_separator)

    return renamed_df


def ect_output_performance_by_ppr(
    eval_df: pd.DataFrame,
    eval_dir: str | None = None,
    positive_rates: Sequence[float] = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5],
) -> pd.DataFrame | None:
    eval_dataset = _df_to_eval_dataset(eval_df)

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset, positive_rates=positive_rates
    )

    df["Total number of unique outcome events"] = _get_num_of_unique_outcome_events(
        eval_dataset=eval_dataset
    )

    df["Number of positive outcomes in test set (TP+FN)"] = (
        df["true_positives"] + df["false_negatives"]
    )

    df = clean_up_performance_by_ppr(df)

    if eval_dir:
        (Path(eval_dir) / "tables").mkdir(parents=True, exist_ok=True)
        df.to_excel(Path(eval_dir) / "tables" / "performance_by_ppr_test.xlsx", index=False)
        return None

    return df


if __name__ == "__main__":
    experiment = "ECT-structured_text-xgboost"
    experiment_path = f"E:/shared_resources/ect/eval_runs/{experiment}_evaluated_on_test"
    flattend_df_dir = "E:/shared_resources/ect/feature_set/flattened_datasets/ect_feature_df/ect_feature_df.parquet"
    outcome_timestamp_col_name = "outc_ect_value_within_0_to_60_days_max_fallback_0"

    ect_output_performance_by_ppr(
        expand_eval_df_with_extra_cols(read_eval_df_from_disk(experiment_path)), experiment_path
    )
