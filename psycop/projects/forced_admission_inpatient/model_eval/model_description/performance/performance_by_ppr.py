import pandas as pd
from wasabi import Printer

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
    get_true_positives,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)

msg = Printer(timestamp=True)


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


def _get_number_of_outcome_events_with_at_least_one_true_positve(
    eval_dataset: EvalDataset, positive_rate: float
) -> float:
    """Get number of outcomes with a prediction time that has at least one true positive prediction.

    Args:
        eval_dataset (EvalDataset): EvalDataset object.
        positive_rate (float, optional): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.

    Returns:
        float: Number of outcomes with at least one true positive.
    """

    # Generate df with only true positives
    tp_df = get_true_positives(eval_dataset=eval_dataset, positive_rate=positive_rate)

    # Return number of outcome events with at least one true positives
    return tp_df[["id", "outcome_timestamps"]].drop_duplicates().shape[0]


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

    renamed_df["Median days from first positive to outcome"] = round(df["median_warning_days"], 1)

    return renamed_df


def fa_inpatient_output_performance_by_ppr(run: ForcedAdmissionInpatientPipelineRun):
    output_path = (
        run.paper_outputs.paths.tables / run.paper_outputs.artifact_names.performance_by_ppr
    )
    positive_rates = [0.5, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01]
    eval_dataset = run.pipeline_outputs.get_eval_dataset()

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset, positive_rates=positive_rates
    )

    df["Total number of unique outcome events"] = _get_num_of_unique_outcome_events(
        eval_dataset=eval_dataset
    )

    df["Number of positive outcomes in test set (TP+FN)"] = (
        df["true_positives"] + df["false_negatives"]
    )

    df["Number of unique outcome events detected ≥1"] = [
        _get_number_of_outcome_events_with_at_least_one_true_positve(
            eval_dataset=eval_dataset, positive_rate=pos_rate
        )
        for pos_rate in positive_rates
    ]

    df["Prop. of unique outcome events detected ≥1"] = round(
        df["Number of unique outcome events detected ≥1"]
        / df["Total number of unique outcome events"],
        3,
    )

    df["Benefit/harm value"] = (df["true_positives"] * 10) - df["false_positives"]

    df["Benefit/harm value based on unique outcomes detected ≥1"] = (
        df["Number of unique outcome events detected ≥1"] * 10
    ) - df["false_positives"]

    df = clean_up_performance_by_ppr(df)

    df.to_excel(output_path, index=False)


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_inpatient_output_performance_by_ppr(run=get_best_eval_pipeline())
