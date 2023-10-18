import pandas as pd
from wasabi import Printer

from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    PipelineRun,
)

msg = Printer(timestamp=True)


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
            "prop of all events captured": "% of all FA captured",
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

    renamed_df["Median days from first positive to FA"] = round(
        df["median_warning_days"],
        1,
    )

    return renamed_df


def fa_output_performance_by_ppr(run: PipelineRun):
    output_path = run.paper_outputs.paths.tables / "performance_by_ppr.xlsx"
    eval_dataset = run.pipeline_outputs.get_eval_dataset()

    df: pd.DataFrame = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_dataset,
        positive_rates=[0.05, 0.04, 0.03, 0.02, 0.01],
    )

    df = clean_up_performance_by_ppr(df)
    df.to_excel(output_path, index=False)


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    fa_output_performance_by_ppr(run=get_best_eval_pipeline)
