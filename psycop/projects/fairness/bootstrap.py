import numpy as np
import pandas as pd


def cluster_bootstrap(
    df: pd.DataFrame,
    rng: np.random.Generator,
    sampling_unit_col: str | list[str] = "dw_ek_borger",
    stratify_col: str | list[str] | None = None,
    sample_weight: bool = False,
) -> pd.DataFrame:
    if stratify_col is not None:
        cluster_df = df[[sampling_unit_col, stratify_col]].drop_duplicates()
    else:
        cluster_df = df[[sampling_unit_col]].drop_duplicates()

    if stratify_col is not None:
        sampled_ids = []
        for _, group in cluster_df.groupby(stratify_col):
            sampled_ids.append(rng.choice(group[sampling_unit_col], size=len(group), replace=True))
        sampled_ids = np.concatenate(sampled_ids).flatten()
    else:
        sampled_ids = rng.choice(cluster_df[sampling_unit_col], size=len(cluster_df), replace=True)

    sampled_clusters = pd.DataFrame(
        {sampling_unit_col: sampled_ids, "_bootstrap_id": np.arange(len(sampled_ids))}
    )

    sampled_clusters = sampled_clusters.merge(df, on=sampling_unit_col)

    if sample_weight:
        sampled_clusters["sample_weight"] = 1 / sampled_clusters.groupby("_bootstrap_id")[
            "timestamp"
        ].transform("count")

    return sampled_clusters


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "dw_ek_borger": [1, 1, 2, 2, 3, 3],
            "y": [0, 1, 0, 1, 0, 1],
            "y_hat": [0, 1, 0, 1, 0, 1],
            "sex": ["M", "M", "F", "F", "M", "M"],
        }
    )

    rng = np.random.default_rng(42)

    cluster_bootstrap(df, rng=rng, sampling_unit_col="dw_ek_borger")
