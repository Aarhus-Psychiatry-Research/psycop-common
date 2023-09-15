"""Load and compare train and validation splits."""
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import plotnine as pn

from psycop.common.model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)

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


def _plot_timestamp_distribution(
    df: pd.DataFrame,
    timestamp_col: str,
    split: str,
    save: bool = True,
) -> None:
    """Plot distribution of timestamps in a column."""

    # Convert the column containing timestamps to a pandas datetime object if it's not already
    if not pd.api.types.is_datetime64_ns_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    df["timestamp_bins"] = pd.cut(
        df[timestamp_col],
        bins=pd.date_range(
            start=df[timestamp_col].min(),
            end=df[timestamp_col].max(),
            freq="10D",
        ),  # type: ignore
    )

    # Create a plot using plotnine
    p = (
        pn.ggplot(df, pn.aes(x="timestamp_bins", fill="timestamp_bins"))
        + pn.geom_bar()
        + pn.labs(
            title="Timestamp Distribution (Binned into 10-Day Intervals)",
            x="Timestamp Intervals",
            y="Count",
        )
        + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
    )

    print(p)

    if save:
        output_dir = (
            Path("E:\\shared_resources\\forced_admissions_inpatient")
            / "data_split_comparisons"
        )

        p.save(output_dir / f"{split}_{timestamp_col}_distribution.png")


def _generate_general_descriptive_stats(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Generate descriptive stats table."""

    unique_patients = df["dw_ek_borger"].nunique()
    mean_entries_per_patient = df.shape[0] / unique_patients
    mean_age = df["pred_age_in_years"].mean()

    female_ratio = df["pred_sex"].sum() / df.shape[0]

    pred_cols = _get_pred_cols_df(df)
    na_ratios = pred_cols.isna().sum().sum() / pred_cols.shape[0] * pred_cols.shape[1]

    outcomes = df["outc_t2d_within_30_days_maximum_fallback_0_dichotomous"].sum()
    outcome_rate = outcomes / df.shape[0]

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
            "na_ratios": [na_ratios],
        },
    )

    return stats


def _get_pred_cols_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get columns that start with pred_."""
    pred_cols = [col for col in df.columns if col.startswith("pred_")]

    return df[pred_cols]


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(
    cfg: FullConfigSchema,
    synth_data: bool = True,
    save: bool = False,
) -> pd.DataFrame:
    """Main."""
    cfg = convert_omegaconf_to_pydantic_object(cfg)  # type: ignore

    if sys.platform == "win32":
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)

    if synth_data:
        df = pd.read_csv(SYNTH_DATA_PATH, index_col=0)

        df["pred_age_in_years"] = np.random.randint(18, 100, df.shape[0])
        df["pred_sex"] = np.random.randint(0, 2, df.shape[0])

        df = df.rename(columns={"citizen_ids": "dw_ek_borger"})

        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

    else:
        train_df = load_and_filter_split_from_cfg(
            data_cfg=cfg.data,
            pre_split_cfg=cfg.preprocessing.pre_split,
            split="train",
        )

        val_df = load_and_filter_split_from_cfg(
            data_cfg=cfg.data,
            pre_split_cfg=cfg.preprocessing.pre_split,
            split="train",
        )

    train_stats = _generate_general_descriptive_stats(train_df, split_name="train")
    val_stats = _generate_general_descriptive_stats(val_df, split_name="val")

    stats = pd.concat([train_stats, val_stats], axis=0).reset_index(drop=True)

    if save:
        output_dir = (
            Path("E:\\shared_resources\\forced_admissions_inpatient")
            / "data_split_comparisons"
        )

        stats.to_csv(output_dir / "train_val_descriptive_comparison.csv")

    _plot_timestamp_distribution(train_df, "timestamp", "train", save)
    _plot_timestamp_distribution(val_df, "timestamp", "val", save)

    return stats


if __name__ == "__main__":
    main()
    print("Done!")
