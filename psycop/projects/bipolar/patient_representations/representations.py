from typing import Literal, Optional

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.bipolar.feature_generation.inspect_feature_sets import load_bp_feature_set
from psycop.projects.bipolar.synthetic_data.bp_synthetic_data import bp_synthetic_data


def _fit_pca(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """Perform PCA on the given DataFrame."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    return pd.DataFrame(components, columns=[f"component_{i+1}" for i in range(n_components)])


def _fit_tsne(df: pd.DataFrame, n_components: int, perplexity: Optional[int]) -> pd.DataFrame:
    """Perform t-SNE on the given DataFrame."""
    perplexity = 20 if perplexity is None else perplexity

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    components = tsne.fit_transform(df)
    return pd.DataFrame(components, columns=[f"component_{i+1}" for i in range(n_components)])


def perform_projection(
    df: pd.DataFrame,
    projecton_algortithm: Literal["pca", "tsne"],
    n_components: int = 2,
    perplexity: Optional[int] = 20,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    # Convert NAs to 0s
    df = df.fillna(0)

    # Limit rows if n_rows is specified (only relevant for t-SNE to avoid high computational cost)
    if n_rows is not None:
        df = df.iloc[:n_rows]

    # Select columns that start with 'pred_'
    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    df_filtered = df[pred_cols]

    # Choose and apply the projecton_algortithm using the appropriate function
    match projecton_algortithm:
        case "pca":
            components_df = _fit_pca(df_filtered, n_components)
        case "tsne":
            components_df = _fit_tsne(df_filtered, n_components, perplexity)

    # Append computed components to the original DataFrame
    for i in range(n_components):
        df[f"component_{i+1}"] = components_df[f"component_{i+1}"].to_numpy()

    return df


if __name__ == "__main__":
    # Load eval data
    best_experiment = "bipolar_model_training_full_feature_v2"
    eval_data = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=best_experiment, metric="all_oof_BinaryAUROC")
        .eval_frame()
        .frame.to_pandas()
    )

    # Rename pred_time_uuid to prediction_time_uuid
    eval_data = eval_data.rename(columns={"pred_time_uuid": "prediction_time_uuid"})

    # Load flattened df
    df = load_bp_feature_set("bipolar_full_feature_set_interval_days_150")

    # Convert df to pandas
    df = df.to_pandas()

    # Merge df onto eval_data on prediction_time_uuid
    df = eval_data.merge(df, on="prediction_time_uuid", how="left")

    pca_df = perform_projection(df, projecton_algortithm="pca")

    ### SYNTHETIC DATA ###
    # Generate synthetic data
    synthetic_df = bp_synthetic_data(num_patients=100)

    synth_pc_df = perform_projection(synthetic_df, projecton_algortithm="pca")