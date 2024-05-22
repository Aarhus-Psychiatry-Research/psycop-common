


from collections.abc import Sequence

from pandas import Index
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.projects.scz_bp.evaluation.configs import SCZ_BP_EVAL_OUTPUT_DIR
from psycop.projects.scz_bp.evaluation.figure2.first_positive_prediction_to_outcome import scz_bp_first_pred_to_event_stratified
from psycop.projects.scz_bp.evaluation.scz_bp_run_evaluation_suite import scz_bp_get_eval_ds_from_best_run_in_experiment
from psycop.common.model_evaluation.binary.performance_by_ppr.performance_by_ppr import (
    generate_performance_by_ppr_table,
)
import pandas as pd

def _format_with_thousand_separator(num: int) -> str:
    return f"{num:,.0f}"


def _format_prop_as_percent(num: float) -> str:
    output = f"{num:.1%}"

    return output

def _clean_up_performance_by_ppr(table: pd.DataFrame) -> pd.DataFrame:
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
            "prop of all events captured": "% of all BP or SCZ captured",
            "f1": "F1",
            "mcc": "MCC",
            "SCZ" : "Median years from first positive to first SCZ diagnosis",
            "BP" : "Median years from first positive to first BP diagnosis",
        },
        axis=1,
    )

    # Handle proportion columns
    prop_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "float64" and c not in ["SCZ", "BP"]]
    for c in prop_cols:
        renamed_df[c] = renamed_df[c].apply(_format_prop_as_percent)

    # Handle count columns
    count_cols = [c for c in renamed_df.columns if renamed_df[c].dtype == "int64"]
    for col in count_cols:
        renamed_df[col] = renamed_df[col].apply(_format_with_thousand_separator)

    renamed_df["Median years from first positive to first SCZ or BP diagnosis"] = round(
        df["median_warning_days"] / 365.25, 1
    )
    renamed_df["Median years from first positive to first SCZ diagnosis"] = round(
        df["SCZ"], 1)
    renamed_df["Median years from first positive to first BP diagnosis"] = round(
        df["BP"], 1)
    

    return renamed_df

def median_years_to_scz_and_bp_by_ppr(eval_ds: EvalDataset, positive_rates: Sequence[float]) -> pd.DataFrame:
    tables = []
    for positive_rate in positive_rates:
        plot_df_with_annotations = scz_bp_first_pred_to_event_stratified(eval_ds=eval_ds, ppr=positive_rate)
        tables.append(pd.DataFrame(plot_df_with_annotations.annotation_dict, index=Index(name="positive_rate", data=[positive_rate])))
    return pd.concat(tables).reset_index()


if __name__ == "__main__":
    best_experiment = "sczbp/test_tfidf_1000"
    positive_rates = [0.08, 0.06, 0.04, 0.02, 0.01]
    eval_ds = scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name=best_experiment)

    df = generate_performance_by_ppr_table(  # type: ignore
        eval_dataset=eval_ds, positive_rates=positive_rates
    )
    median_years_to_scz_bp = median_years_to_scz_and_bp_by_ppr(eval_ds=eval_ds, positive_rates=positive_rates)
    df["positive_rate"] = df["positive_rate"].round(2)


    df = _clean_up_performance_by_ppr(df.merge(median_years_to_scz_bp, on="positive_rate"))
    
    with (SCZ_BP_EVAL_OUTPUT_DIR / "table2.html").open("w") as f:
        f.write(df.to_html())