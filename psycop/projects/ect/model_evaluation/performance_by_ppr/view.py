import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
from psycop.projects.ect.model_evaluation.performance_by_ppr.model import (
    PerformanceByPPRModel,
    performance_by_ppr_model,
)
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk


def performance_by_ppr_view(model: PerformanceByPPRModel, outcome_label: str) -> pl.DataFrame:
    model2pretty = {
        "positive_rate": "Predicted positive rate",
        "true_prevalence": "True prevalence",
        "PPV": "PPV",
        "NPV": "NPV",
        "FPR": "FPR",
        "FNR": "FNR",
        "sensitivity": "Sens",
        "specificity": "Spec",
        "accuracy": "Acc",
        "true_positives": "TP",
        "true_negatives": "TN",
        "false_positives": "FP",
        "false_negatives": "FN",
        "prop_of_all_events_captured": f"% of all {outcome_label} captured",
        "f1": "F1",
    }

    renamed_df = model.rename(model2pretty).select(model2pretty.values())

    # Handle proportion columns
    prop_cols = [col for col, dtype in renamed_df.schema.items() if dtype == pl.Float64]
    for c in prop_cols:
        renamed_df = renamed_df.with_columns([pl.col(c).apply(lambda x: f"{x:.1%}").alias(c)])

    # Handle count columns
    count_cols = [col for col, dtype in renamed_df.schema.items() if dtype == pl.Int64]
    for col in count_cols:
        renamed_df = renamed_df.with_columns(pl.col(col).apply(lambda x: f"{x:,}").alias(col))

    renamed_df = renamed_df.with_columns(
        model.select(
            (pl.col("mean_warning_days"))
            .round(1)
            .alias(f"Mean days from first positive to {outcome_label}")
        )
    ).with_columns(
        model.select(
            pl.col("median_warning_days").alias(
                f"Median days from first positive to {outcome_label}"
            )
        )
    )

    return renamed_df


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    structured_only_experiment = "ECT-structured_text-xgboost"
    structured_only_experiment_path = (
        f"E:/shared_resources/ect/eval_runs/{structured_only_experiment}_evaluated_on_test"
    )
    structured_only_df = read_eval_df_from_disk(structured_only_experiment_path)
    eval_df = EvalFrame(frame=structured_only_df, allow_extra_columns=True)

    table = performance_by_ppr_view(
        performance_by_ppr_model(
            eval_df=eval_df, positive_rates=[0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.5]
        ),
        outcome_label="ECT",
    )
    table.write_excel(f"{structured_only_experiment_path}/performance_by_ppr.xlsx")
