from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid

data_path = OVARTACI_SHARED_DIR / "uti" / "flattened_datasets" / "flattened_datasets"
output_path = OVARTACI_SHARED_DIR / "uti" / "data_visualisations"

np.random.seed(42)

# Parameters for the synthetic dataset
n_rows = 2000  # Number of rows
patient_ids = np.random.randint(1, 101, size=n_rows)  # 100 unique patients
timestamps = pd.date_range(start="2011-01-01", end="2021-12-31", periods=n_rows)
values = np.random.choice(
    [0, 66001, 660123, 60015, 60024], size=n_rows, p=[0.5, 0.125, 0.125, 0.125, 0.125]
)

data = pd.DataFrame({"timestamp": timestamps, "id": patient_ids, "out_value": values})


def plot_prediction_times_by_year(df: pd.DataFrame) -> pn.ggplot:
    # extract the year from the timestamp
    df["year"] = df["timestamp"].dt.year

    # plot the number of prediction times by year as a bar plot
    plot = (
        pn.ggplot(df, pn.aes(x="year", fill="year"))
        + pn.geom_bar(fill="#1f77b4")  # Dark blue color
        + pn.labs(title="Predictions pr. år", x="Year", y="Number of prediction times")
        + pn.theme_classic()  # Modern theme
        + pn.theme(
            text=pn.element_text(size=12),
            axis_title=pn.element_text(size=14),
            title=pn.element_text(size=16),
        )
    )

    return plot


def plot_outcomes_by_year(
    df: pd.DataFrame, outcome_col_name: str = "out_value", outcome_def: str = "UTIs"
) -> pn.ggplot:
    df[outcome_col_name] = df[outcome_col_name].replace("Ukendt", "1111")
    df[outcome_col_name] = pd.to_numeric(df[outcome_col_name], errors="coerce")

    # keep only outcomes that are not 0
    df = df[df[outcome_col_name] != 0]

    df["year"] = df["timestamp"].dt.year

    # plot the number of prediction times by year as a bar plot
    plot = (
        pn.ggplot(df, pn.aes(x="year", fill="year"))
        + pn.geom_bar(fill="#1f77b4")  # Dark blue color
        + pn.labs(title=f"UTI-outcomes pr. år {outcome_def}", x="Year", y="Number of outcomes")
        + pn.theme_classic()  # Modern theme
        + pn.theme(
            text=pn.element_text(size=12),
            axis_title=pn.element_text(size=14),
            title=pn.element_text(size=16),
        )
        + pn.annotate(
            "text",
            x=0.9,
            y=0.9,
            label=f"Dataset length: {len(df)}",
            ha="left",
            va="top",
            size=10,
            color="red",
        )
    )

    return plot


def plot_outcomes_by_shak_code(df: pd.DataFrame, outcome_col_name: str = "out_value") -> pn.ggplot:
    df[outcome_col_name] = df[outcome_col_name].replace("Ukendt", "1111")
    df[outcome_col_name] = pd.to_numeric(df[outcome_col_name], errors="coerce")

    # keep only outcomes that are not 0
    df = df[df[outcome_col_name] != 0]

    # keep only the first four digits of the outcome code
    df["shak_code"] = df[outcome_col_name].astype(str).str[:4]

    # delete rows with nan values in the shake code column
    df = df[df["shak_code"] != "nan"]

    counts = df["shak_code"].value_counts()
    df_filtered = df[df["shak_code"].isin(counts[counts >= 5].index)]

    # plot the number of prediction times by shak code as a bar plot
    plot = (
        pn.ggplot(df_filtered, pn.aes(x="shak_code", fill="shak_code"))
        + pn.geom_bar(fill="#1f77b4")  # Dark blue color
        + pn.labs(title="UTI-udfald pr. afdeling", x="Shak code", y="Number of outomces")
        + pn.theme_classic()  # Modern theme
        + pn.theme(
            text=pn.element_text(size=12),
            axis_title=pn.element_text(size=14),
            title=pn.element_text(size=16),
        )
    )

    return plot


def plot_outcomes_gender(df: pd.DataFrame, outcome_col_name: str = "out_value") -> pn.ggplot:
    df[outcome_col_name] = df[outcome_col_name].replace("Ukendt", "1111")
    df[outcome_col_name] = pd.to_numeric(df[outcome_col_name], errors="coerce")

    # keep only outcomes that are not 0
    df = df[df[outcome_col_name] != 0]

    df["pred_sex_female_fallback_0"] = df["pred_sex_female_fallback_0"].replace(
        {0: "Male", 1: "Female"}
    )

    # plot the number of prediction times by shak code as a bar plot
    plot = (
        pn.ggplot(df, pn.aes(x="pred_sex_female_fallback_0", fill="pred_sex_female_fallback_0"))
        + pn.geom_bar(fill="#1f77b4")  # Dark blue color
        + pn.labs(
            title="UTI-outcome - sex distribution", x="Day of admissions", y="Number of outcomes"
        )
        + pn.theme_classic()  # Modern theme
        + pn.theme(
            text=pn.element_text(size=12),
            axis_title=pn.element_text(size=14),
            title=pn.element_text(size=16),
        )
    )

    return plot


def plot_outcomes_adm_day(df: pd.DataFrame, outcome_col_name: str = "out_value") -> pn.ggplot:
    df[outcome_col_name] = df[outcome_col_name].replace("Ukendt", "1111")
    df[outcome_col_name] = pd.to_numeric(df[outcome_col_name], errors="coerce")

    # keep only outcomes that are not 0
    df = df[df[outcome_col_name] != 0]

    adm_day_counts = df["pred_adm_day_count"].value_counts()
    df_filtered = df[df["pred_adm_day_count"].isin(adm_day_counts[adm_day_counts >= 5].index)]

    # plot the number of prediction times by shak code as a bar plot
    plot = (
        pn.ggplot(df_filtered, pn.aes(x="pred_adm_day_count", fill="pred_adm_day_count"))
        + pn.geom_bar(fill="#1f77b4")  # Dark blue color
        + pn.labs(
            title="UTI-outcome - admission day  distribution",
            x="Gender",
            y="Number of outcomes times",
        )
        + pn.theme_classic()  # Modern theme
        + pn.theme(
            text=pn.element_text(size=12),
            axis_title=pn.element_text(size=14),
            title=pn.element_text(size=16),
        )
    )

    return plot


def uti_patchwork_data_plot(
    df: pd.DataFrame,
    output_path: Path,
    name: str = "uti_data_pathwork",
    outcome_col_name: str = "outc_uti_value_within_0_to_1_days_max_fallback_0",
    outcome_def: str = "UTIs",
) -> None:
    outc_per_year = plot_outcomes_by_year(df, outcome_col_name, outcome_def)
    outc_per_shak = plot_outcomes_by_shak_code(df, outcome_col_name)
    outc_gender = plot_outcomes_gender(df, outcome_col_name)
    outc_adm_day = plot_outcomes_adm_day(df, outcome_col_name)

    plots = [outc_gender, outc_adm_day, outc_per_year, outc_per_shak]

    grid = create_patchwork_grid(plots=plots, single_plot_dimensions=(5, 3), n_in_row=2)
    grid_output_path = output_path / f"{name}.png"
    grid.savefig(grid_output_path)


if __name__ == "__main__":
    full_definition = pd.read_parquet(
        data_path / "uti_outcomes_full_definition" / "uti_outcomes_full_definition.parquet"
    )

    uti_patchwork_data_plot(
        full_definition, output_path, name="full_definition", outcome_def="Full definition"
    )

    urine_samples = pd.read_parquet(
        data_path / "uti_outcomes_urine_samples" / "uti_outcomes_urine_samples.parquet"
    )

    uti_patchwork_data_plot(
        urine_samples, output_path, name="urine_amples", outcome_def="Positive urine samples"
    )

    antibiotics = pd.read_parquet(
        data_path / "uti_outcomes_antibiotics" / "uti_outcomes_antibiotics.parquet"
    )

    uti_patchwork_data_plot(antibiotics, output_path, name="antibiotics", outcome_def="Antibiotics")

    plot_prediction_times_by_year(full_definition).save(
        output_path / "prediction_times_by_year.png"
    )
