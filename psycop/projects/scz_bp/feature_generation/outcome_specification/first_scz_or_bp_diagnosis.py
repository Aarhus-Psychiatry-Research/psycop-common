import pandas as pd
import polars as pl

from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_eligible_config import (
    N_DAYS_WASHIN,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.add_time_from_first_visit import (
    add_time_from_first_visit,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.bp_diagnoses import (
    get_first_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.scz_diagnoses import (
    get_first_scz_diagnosis,
)


def get_first_scz_or_bp_diagnosis() -> pl.DataFrame:
    dfs = {"scz": get_first_scz_diagnosis(), "bp": get_first_bp_diagnosis()}
    for df_type in dfs:
        dfs[df_type]["source"] = df_type

    combined = pd.concat(
        [dfs[k] for k in dfs],
        axis=0,
    )

    first_scz_or_bp_indicator = (
        combined.sort_values("timestamp")
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )
    # subtract 1 day from first diagnosis to avoid any leakage and add indicator column
    first_scz_or_bp_indicator = pl.from_pandas(first_scz_or_bp_indicator).with_columns(
        pl.col("timestamp") - pl.duration(days=1),
        pl.lit(1).alias("value")
    )
    return first_scz_or_bp_indicator


def get_first_scz_or_bp_diagnosis_with_time_from_first_visit() -> pl.DataFrame:
    first_scz_or_bp = get_first_scz_or_bp_diagnosis()
    first_scz_or_bp = add_time_from_first_visit(df=first_scz_or_bp)
    return first_scz_or_bp


def get_scz_bp_patients_excluded_by_washin() -> pl.Series:
    first_scz_or_bp = get_first_scz_or_bp_diagnosis_with_time_from_first_visit()

    return first_scz_or_bp.filter(
        pl.col("time_from_first_visit") < pl.duration(days=N_DAYS_WASHIN)
    ).get_column("dw_ek_borger")


def get_first_scz_bp_diagnosis_after_washin() -> pl.DataFrame:
    first_scz_or_bp = get_first_scz_or_bp_diagnosis_with_time_from_first_visit()

    return first_scz_or_bp.filter(
        pl.col("time_from_first_visit") >= pl.duration(days=N_DAYS_WASHIN)
    ).select("dw_ek_borger", "timestamp", "value", "source")


if __name__ == "__main__":
    df = get_first_scz_bp_diagnosis_after_washin()
    excluded = get_scz_bp_patients_excluded_by_washin()
