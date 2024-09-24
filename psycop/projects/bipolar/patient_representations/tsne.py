import pandas as pd
from sklearn.manifold import TSNE

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.bipolar.feature_generation.inspect_feature_sets import load_bp_feature_set
from psycop.projects.bipolar.synthetic_data.bp_synthetic_data import bp_synthetic_data


def perform_tsne(
    df: pd.DataFrame, n_components: int = 2, perplexity: int = 30, n_iter: int = 1000
) -> pd.DataFrame:
    # Convert NAs to 0s
    df = df.fillna(0)

    # Drop columns that don't start with 'pred_'
    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    df_filtered = df[pred_cols]

    # Perform t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    components = tsne.fit_transform(df_filtered)

    tsne_df = pd.DataFrame(components, columns=["component_1", "component_2"])

    # Append t-SNE components to the original dataframe
    df["component_1"] = tsne_df["component_1"].to_numpy()
    df["component_2"] = tsne_df["component_2"].to_numpy()

    return df


if __name__ == "__main__":
    # Load eval data
    best_experiment = "bipolar_test"
    eval_data = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=best_experiment, metric="all_oof_BinaryAUROC")
        .eval_frame()
        .frame.to_pandas()
    )

    # Rename pred_time_uuid to prediction_time_uuid
    eval_data = eval_data.rename(columns={"pred_time_uuid": "prediction_time_uuid"})

    # Load flattened df
    df = load_bp_feature_set("structured_predictors_2_layer_interval_days_100")

    # Convert df to pandas
    df = df.to_pandas()

    # Merge df onto eval_data on prediction_time_uuid
    df = eval_data.merge(df, on="prediction_time_uuid", how="left")

    ### SYNTHETIC DATA ###
    # Generate synthetic data
    synthetic_df = bp_synthetic_data(num_patients=100)

    # Perform t-SNE
    tsne_df = perform_tsne(synthetic_df)

    # Display the first few rows of the DataFrame with t-SNE components
    print(tsne_df.head())
