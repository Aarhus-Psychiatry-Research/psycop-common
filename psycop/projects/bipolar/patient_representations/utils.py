import pandas as pd
from sklearn.decomposition import PCA

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.bipolar.feature_generation.inspect_feature_sets import (
    load_bp_feature_set,
)


def prepare_eval_data_for_pca() -> pd.DataFrame:
    # Load eval data
    best_experiment = "bipolar_full_test"
    eval_data = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=best_experiment, metric="all_oof_BinaryAUROC")
        .eval_frame()
        .frame.to_pandas()
    )

    # rename pred_time to prediction_time_uuid
    eval_data = eval_data.rename(columns={"pred_time_uuid": "prediction_time_uuid"})

    # Load flattened df
    df = load_bp_feature_set("structured_predictors_2_layer_interval_days_100")

    # convert df to pandas
    df = df.to_pandas()

    # merge df onto eval_data on prediction_time_uuid
    df = eval_data.merge(df, on="prediction_time_uuid", how="left")

    df["prediction"] = get_predictions_for_positive_rate(
        y_hat_probs=df["y_hat_prob"], desired_positive_rate=0.05
    )[0]

    df["prediction"] = df["prediction"].astype(int)
    df["y"] = df["y"].astype(int)
    df["prediction_type"] = None
    df.loc[(df["prediction"] == 1) & (df["y"] == 1), "prediction_type"] = "TP"
    df.loc[(df["prediction"] == 1) & (df["y"] == 0), "prediction_type"] = "FP"
    df.loc[(df["prediction"] == 0) & (df["y"] == 0), "prediction_type"] = "TN"
    df.loc[(df["prediction"] == 0) & (df["y"] == 1), "prediction_type"] = "FN"

    return df


if __name__ == "__main__":
    df = prepare_eval_data_for_pca()
