"""Load and compare train and validation splits."""

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.utils import load_and_filter_split_from_cfg
from psycop.projects.forced_admission_inpatient.model_eval.config import (
    DEV_GROUP_NAME,
    DEVELOPMENT_GROUP,
    PROJECT_MODEL_DIR,
)

model_name = DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[0, 2]

PRETRAINED_CFG_PATH = PROJECT_MODEL_DIR / DEV_GROUP_NAME / model_name / "cfg.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "model_training" / "config"
REPO_ROOT = Path(__file__).resolve().parents[4]
SYNTH_DATA_PATH = (
    REPO_ROOT
    / "common"
    / "test_utils"
    / "test_data"
    / "flattened"
    / "synth_flattened_with_outcome.csv"
)


def _get_cfg_from_json_file(path: Path) -> FullConfigSchema:
    # Loading the json instead of the .pkl makes us independent
    # of whether the imports in psycop-common model-training have changed
    # TODO: Note that this means assigning to the cfg property does nothing, since it's recomputed every time it's called
    return FullConfigSchema.model_validate(json.loads(json.loads(path.read_text())))


def plot_timestamp_distribution(
    df: pd.DataFrame, timestamp_column: str, split_name: str, output_dir: Optional[Path]
):
    # Convert the column to a pandas datetime object if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Calculate the bins using 10-day intervals
    min_date = df[timestamp_column].min()
    max_date = df[timestamp_column].max()
    bins = pd.date_range(start=min_date, end=max_date, freq="10D")

    # Create a histogram of the timestamps
    plt.figure(figsize=(10, 6))
    plt.hist(df[timestamp_column], bins=bins, edgecolor="k", alpha=0.7)  # type: ignore

    # Set plot labels and title
    plt.xlabel("Timestamp")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {timestamp_column} in 10-Day Intervals")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot as an image if save_path is provided
    if output_dir:
        plt.savefig(output_dir / f"{split_name}_timestamp_distribution.png")


def _generate_general_descriptive_stats(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Generate descriptive stats table."""

    unique_patients = df["dw_ek_borger"].nunique()
    mean_entries_per_patient = df.shape[0] / unique_patients
    mean_age = df["pred_age_in_years"].mean()

    female_ratio = df["pred_sex_female"].sum() / df.shape[0]

    pred_cols = _get_pred_cols_df(df)
    na_ratios = pred_cols.isna().sum().sum() / (pred_cols.shape[0] * pred_cols.shape[1])

    outcomes = df["outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"].sum()
    outcome_rate = outcomes / df.shape[0]

    unique_patients_with_outcomes = df[
        df["outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"] == 1
    ]["dw_ek_borger"].nunique()

    print(
        df[df["outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"] == 1][
            "dw_ek_borger"
        ]
        .value_counts()
        .head(10)
    )

    stats = pd.DataFrame(
        data={
            "split_name": [split_name],
            "number_of_entries": [df.shape[0]],
            "unique_patients": [unique_patients],
            "mean_entries_per_patient": [mean_entries_per_patient],
            "mean_age": [mean_age],
            "female_ratio": [female_ratio],
            "outcome_rate": [outcome_rate],
            "outcomes": [outcomes],
            "unique_patients_with_outcomes": [unique_patients_with_outcomes],
            "na_ratios": [na_ratios],
        }
    )

    return stats


def _get_pred_cols_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get columns that start with pred_."""
    pred_cols = [col for col in df.columns if col.startswith("pred_")]  # type: ignore

    return df[pred_cols]


def _calc_feature_corr_with_outcome(
    train_df: pd.DataFrame, val_df: pd.DataFrame, only_tfidf: bool = False
) -> pd.DataFrame:
    """
    Calculate the Pearson correlation coefficients between numeric feature columns$
    and a binary outcome column.
    """
    print("Calculating correlations!")

    if only_tfidf:
        pred_cols = [col for col in train_df.columns if col.startswith("pred_pred_tfidf")]  # type: ignore

        train_correlations = train_df[
            [*pred_cols, "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"]
        ].corr()[
            "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"
        ]  # Pearsonns corr coeff
        val_correlations = val_df[
            [*pred_cols, "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"]
        ].corr()[
            "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"
        ]  # Pearsonns corr coeff

    else:
        pred_cols = [col for col in train_df.columns if col.startswith("pred_")]  # type: ignore

        train_correlations = train_df[
            [*pred_cols, "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"]
        ].corr()[
            "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"
        ]  # Pearsonns corr coeff
        val_correlations = val_df[
            [*pred_cols, "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"]
        ].corr()[
            "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous"
        ]  # Pearsonns corr coeff

    # Create a DataFrame to store feature names and correlations
    result_df = pd.DataFrame(
        {
            "feature": pred_cols,
            "train_split_correlations": train_correlations[pred_cols],  # type: ignore
            "val_split_correlations": val_correlations[pred_cols],  # type: ignore
            "correlation_diffs": train_correlations[pred_cols] - val_correlations[pred_cols],  # type: ignore
        }
    ).reset_index(drop=True)

    return result_df


def _plot_corr_diffs(
    df: pd.DataFrame,
    shape: tuple,  # type: ignore
    output_dir: Optional[Path],
    feature_split: str = "features",
):
    corr_array = df["correlation_diffs"].values.reshape(shape[0], shape[1])  # type: ignore

    # Create a green to red colormap
    cmap = LinearSegmentedColormap.from_list("GreenToRed", ["#FF0000", "#00FF00"], N=256)

    # Create a heatmap of the decimal values
    plt.figure(figsize=(8, 8))
    plt.imshow(corr_array, cmap=cmap, interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(label="Values", orientation="vertical")

    # Add labels to the x and y axes if needed
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Set plot title
    plt.title("Correlation Values Heatmap (Train split - validation split)")

    plt.tight_layout()

    # Save the plot as an image if save_path is provided
    if output_dir:
        plt.savefig(output_dir / f"{feature_split}_correlation_differences.png")


def main(cfg: FullConfigSchema, synth_data: bool = False, save: bool = True) -> pd.DataFrame:
    """Main."""
    output_dir = (
        Path("E:\\shared_resources\\forced_admissions_inpatient") / "data_split_comparisons"
    )

    cfg = _get_cfg_from_json_file(path=PRETRAINED_CFG_PATH)

    if sys.platform == "win32":
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)

    if synth_data:
        df = pd.read_csv(SYNTH_DATA_PATH, index_col=0)

        df["pred_age_in_years"] = np.random.randint(18, 100, df.shape[0])
        df["pred_sex_female"] = np.random.randint(0, 2, df.shape[0])

        df = df.rename(
            columns={
                "citizen_ids": "dw_ek_borger",
                "outc_t2d_within_30_days_maximum_fallback_0_dichotomous": "outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous",
            }
        )

        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)  # type: ignore

    else:
        train_df = load_and_filter_split_from_cfg(
            data_cfg=cfg.data, pre_split_cfg=cfg.preprocessing.pre_split, split="train"
        )

        val_df = load_and_filter_split_from_cfg(
            data_cfg=cfg.data, pre_split_cfg=cfg.preprocessing.pre_split, split="val"
        )

    train_stats = _generate_general_descriptive_stats(train_df, split_name="train")
    val_stats = _generate_general_descriptive_stats(val_df, split_name="val")

    stats = pd.concat([train_stats, val_stats], axis=0).reset_index(drop=True)

    if save:
        stats.to_csv(output_dir / "train_val_descriptive_comparison.csv")

    corrs = _calc_feature_corr_with_outcome(train_df, val_df)

    if save:
        corrs.to_csv(output_dir / "feature_outcome_correlations.csv")

    tfidf_features = corrs[corrs["feature"].str.startswith("pred_pred_tfidf")]
    other_features = corrs[~corrs["feature"].str.startswith("pred_pred_tfidf")]

    _plot_corr_diffs(
        tfidf_features, (25, 30), output_dir=output_dir, feature_split="tfidf_features"
    )
    _plot_corr_diffs(
        other_features, (49, 22), output_dir=output_dir, feature_split="other_features"
    )

    plot_timestamp_distribution(
        train_df, timestamp_column="timestamp", split_name="train", output_dir=output_dir
    )
    plot_timestamp_distribution(
        val_df, timestamp_column="timestamp", split_name="val", output_dir=output_dir
    )

    return stats


if __name__ == "__main__":
    cfg = _get_cfg_from_json_file(path=PRETRAINED_CFG_PATH)
    main(cfg)
    print("Done!")
