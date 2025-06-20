import numpy as np
import pandas as pd
import polars as pl
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female


def read_eval_df_from_disk(experiment_path: str) -> pl.DataFrame:
    return pl.read_parquet(experiment_path + "/eval_df.parquet")

def parse_dw_ek_borger_from_uuid(
    df: pl.DataFrame, output_col_name: str = "dw_ek_borger"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuid").str.split("-").list.first().cast(pl.Int64).alias(output_col_name)
    )

if __name__ == "__main__":
    eval_df = read_eval_df_from_disk("E:/shared_resources/restraint/eval_runs/restraint_all_tuning_v2_best_run_evaluated_on_test")
    sex_df = sex_female()
    
    joint_df = (
        parse_dw_ek_borger_from_uuid(df).join(sex_df, on="dw_ek_borger", how="left")
    ).to_pandas()

    df = pl.DataFrame(
        auroc_by_model(
            input_values=joint_df["sex_female"],
            y=joint_df["y"],
            y_hat_probs=joint_df["y_hat_prob"],
            input_name="sex",
            bin_continuous_input=False,
        )
    ).with_columns(
        pl.when(pl.col("sex")).then(pl.lit("Female")).otherwise(pl.lit("Male")).alias("sex")
    ).to_pandas()

    y_hat = get_predictions_for_positive_rate(0.01, eval_df.y_hat_prob)[0]

    metrics = {
        "AUROC": roc_auc_score,
        "Positive predictive value": precision_score,
        "False positive rate": false_positive_rate,
        "False negative rate": false_negative_rate,
        "Selection rate": selection_rate,
        "Demographic parity": demographic_parity_difference,
        "Equalised odds": equalized_odds_difference,
        "Count": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics, y_true=eval_ds.y, y_pred=y_hat, sensitive_features=eval_ds.is_female.replace({True: "Female", False: "Male"})
    )

    bar_plot = metric_frame.by_group.plot.bar(
        subplots=True, layout=[3, 2], legend=False, figsize=[12, 8], title="Prediction of schizophrenia and bipolar disorder", colormap="Dark2", xlabel="Sex"
    )

    bar_plot[0][0].figure.savefig("bar_plot.png")

    pass