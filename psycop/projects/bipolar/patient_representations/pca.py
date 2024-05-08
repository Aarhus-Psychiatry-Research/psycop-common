import pandas as pd
from sklearn.decomposition import PCA

from psycop.projects.bipolar.synthetic_data.bp_synthetic_data import bp_synthetic_data


def perform_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    # Drop columns that don't start with 'pred_'
    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    df_filtered = df[pred_cols]

    # Perform PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_filtered)

    pca_df = pd.DataFrame(components, columns=["component_1", "component_2"])

    # appende pca_df to df
    pca_df = pd.concat([df, pca_df], axis=1)

    return pca_df


if __name__ == "__main__":
    # Generate synthetic data
    synthetic_df = bp_synthetic_data(num_patients=100)

    # Perform PCA
    pca_df = perform_pca(synthetic_df)

    # Display first few rows of the DataFrame
    print(pca_df.head())
