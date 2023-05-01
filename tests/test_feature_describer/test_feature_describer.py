"""Tests of the feature describer module."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from psycop_feature_generation.data_checks.flattened.feature_describer import (
    generate_feature_description_df,
    generate_feature_description_row,
    save_feature_descriptive_stats_from_dir,
)
from psycop_feature_generation.utils import RELATIVE_PROJECT_ROOT
from timeseriesflattener.feature_spec_objects import (
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)


@pytest.fixture()
def predictor_specs():
    return [
        PredictorSpec(
            values_df=pd.DataFrame({"hba1c": [0]}),
            interval_days=100,
            resolve_multiple_fn="max",
            fallback=np.nan,
            feature_name="hba1c",
            input_col_name_override="hba1c",
        ),
    ]


@pytest.fixture()
def static_specs():
    return [
        StaticSpec(
            values_df=pd.DataFrame({"value": [0]}),
            prefix="pred",
            feature_name="hba1c",
        ),
    ]


@pytest.fixture()
def outcome_specs():
    return [
        OutcomeSpec(
            values_df=pd.DataFrame({"value": [0]}),
            prefix="outc",
            incident="true",
            lookahead_days=30,
            feature_name="t2d",
            resolve_multiple_fn="max",
            fallback="0",
        ),
    ]


@pytest.fixture()
def df():
    """Load the synthetic flattened data set."""
    return pd.read_csv(
        RELATIVE_PROJECT_ROOT
        / "tests/test_data/flattened/generated_with_outcome/synth_flattened_with_outcome.csv",
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
        feature_specs=specs,
        file_suffix="parquet",
        splits=("train",),
        out_dir=RELATIVE_PROJECT_ROOT
        / "tests"
        / "outputs_from_tests"
        / "feature_describer",
    )


def test_load_dataset(df):
    """Check loading of synthetic dataset."""
    assert df.shape[0] == 10_000


def test_generate_feature_description_row_for_temporal_spec(
    df: pd.DataFrame,
    predictor_specs: list[PredictorSpec],
):
    spec = predictor_specs[0]

    column_name = spec.get_col_str()

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

    generate_feature_description_df(df=df, predictor_specs=predictor_specs)


def test_generate_feature_description_row_for_static_spec(
    df: pd.DataFrame,
    static_specs: list[PredictorSpec],
):
    spec = static_specs[0]

    column_name = spec.get_col_str()

    df = df.rename(
        columns={"pred_hba1c_within_100_days_max_fallback_nan": column_name},
    )

    generate_feature_description_row(series=df[column_name], predictor_spec=spec)
