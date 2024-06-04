import pandas as pd
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
)
from psycop.projects.cvd.model_evaluation.single_run.performance_by_ppr.model import (
    PerformanceByPPRModel,
    performance_by_ppr_model,
)


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
            (pl.col("mean_warning_days") / 365.25)
            .round(1)
            .alias(f"Mean years from first positive to {outcome_label}")
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

    eval_df = (
        MlflowClientWrapper()
        .get_run(experiment_name="baseline_v2_cvd", run_name="Layer 1")
        .eval_frame()
    )
    table = performance_by_ppr_view(
        performance_by_ppr_model(
            eval_df=eval_df,
            positive_rates=[0.01, 0.05, 0.1, 0.2],
            outcome_timestamps=cvd_outcome_timestamps(),
        ),
        outcome_label="CVD",
    )
    table.write_csv("test.csv")
