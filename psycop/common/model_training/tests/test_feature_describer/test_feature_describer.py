"""Tests of the feature describer module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from timeseriesflattener.v1.aggregation_fns import maximum
from timeseriesflattener.v1.feature_specs.single_specs import OutcomeSpec, PredictorSpec, StaticSpec

from psycop.common.feature_generation.data_checks.flattened.feature_describer import (
    generate_feature_description_df,
    generate_feature_description_row,
    save_feature_descriptive_stats_from_dir,
)
from psycop.common.global_utils.paths import PSYCOP_PKG_ROOT


@pytest.fixture
def predictor_specs() -> list[PredictorSpec]:
    return [
        PredictorSpec(
            timeseries_df=pd.DataFrame({"hba1c": [0]}),
            lookbehind_days=100,
            aggregation_fn=maximum,
            fallback=np.nan,
            feature_base_name="hba1c",
        )
    ]


@pytest.fixture
def static_specs() -> list[StaticSpec]:
    return [
        StaticSpec(
            timeseries_df=pd.DataFrame({"value": [0]}), prefix="pred", feature_base_name="hba1c"
        )
    ]


@pytest.fixture
def outcome_specs() -> list[OutcomeSpec]:
    return [
        OutcomeSpec(
            timeseries_df=pd.DataFrame({"value": [0]}),
            prefix="outc",
            incident=True,
            lookahead_days=30,
            feature_base_name="t2d",
            aggregation_fn=maximum,
            fallback=0,
        )
    ]


@pytest.fixture
def df() -> pd.DataFrame:
    """Load the synthetic flattened data set."""
    return pd.read_csv(
        PSYCOP_PKG_ROOT
        / "test_utils"
        / "test_data"
        / "flattened"
        / "synth_flattened_with_outcome.csv"
    )


def test_save_feature_descriptive_stats_from_dir(
    tmp_path: Path,
    df: pd.DataFrame,
    predictor_specs: list[PredictorSpec],
    outcome_specs: list[OutcomeSpec],
):
    specs = predictor_specs + outcome_specs

    # Save df to tmp_path
    df.to_parquet(tmp_path / "train.parquet")

    save_feature_descriptive_stats_from_dir(
        feature_set_dir=tmp_path,
        feature_specs=specs,  # type: ignore
        file_suffix="parquet",
        splits=("train",),
        out_dir=tmp_path,
        prefixes_to_describe={"pred"},
    )


def test_load_dataset(df: pd.DataFrame) -> None:
    """Check loading of synthetic dataset."""
    assert df.shape[0] == 10_000


def test_generate_feature_description_row_for_temporal_spec(
    df: pd.DataFrame, predictor_specs: list[PredictorSpec]
):
    spec = predictor_specs[0]

    column_name = spec.get_output_col_name()

    row = generate_feature_description_row(series=df[column_name], predictor_spec=spec)

    Number = (int, float)

    assert isinstance(row["Predictor df"], str)
    assert isinstance(row["Lookbehind days"], Number)
    assert isinstance(row["Resolve multiple"], str)
    assert isinstance(row["N unique"], int)
    assert isinstance(row["Fallback strategy"], str)
    assert isinstance(row["Proportion missing"], Number)
    assert isinstance(row["Mean"], Number)
    assert isinstance(row["Proportion using fallback"], Number)

    generate_feature_description_df(
        df=df,
        predictor_specs=predictor_specs,  # type: ignore
        prefixes_to_describe="pred_",  # type: ignore
    )  # type: ignore


def test_generate_feature_description_row_for_static_spec(
    df: pd.DataFrame, static_specs: list[PredictorSpec]
):
    spec = static_specs[0]

    column_name = spec.get_output_col_name()

    df = df.rename(columns={"pred_hba1c_within_100_days_maximum_fallback_nan": column_name})

    generate_feature_description_row(series=df[column_name], predictor_spec=spec)
