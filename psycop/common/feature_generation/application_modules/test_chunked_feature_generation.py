import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_series_equal
from timeseriesflattener.v1.aggregation_fns import mean
from timeseriesflattener.v1.feature_specs.single_specs import PredictorSpec

from psycop.common.feature_generation.application_modules.chunked_feature_generation import (
    ChunkedFeatureGenerator,
)
from psycop.common.feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop.common.feature_generation.application_modules.project_setup import ColNames, ProjectInfo
from psycop.common.test_utils.str_to_df import str_to_df


@pytest.fixture()
def synth_prediction_times() -> pd.DataFrame:
    return str_to_df(
        """entity_id,timestamp
999,2000-01-02,
1,2020-01-02 00:00:00,
2,2020-01-02 00:00:00,
3,2020-01-02 00:00:00,
4,2020-01-02 00:00:00,
5,2020-01-02 00:00:00,
6,2020-01-02 00:00:00,
999,2030-01-01,
"""
    )


@pytest.fixture()
def synth_predictor_1() -> pd.DataFrame:
    return str_to_df(
        """entity_id,timestamp,value
1,2020-01-01,1
2,2020-01-01,0
3,2020-01-01,1
4,2020-01-01,0
5,2020-01-01,1
6,2020-01-01,0
"""
    )


@pytest.fixture()
def synth_predictor_2() -> pd.DataFrame:
    return str_to_df(
        """entity_id,timestamp,value
1,2020-01-01,0
2,2020-01-01,1
3,2020-01-01,0
4,2020-01-01,1
5,2020-01-01,0
6,2020-01-01,1
"""
    )


@pytest.fixture()
def synth_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="test",
        project_path=Path(tempfile.TemporaryDirectory().name),
        col_names=ColNames(timestamp="timestamp", id="entity_id"),
    )


def test_find_shared_cols():
    dfs = [pl.DataFrame({"a": [1], "b": [4]}), pl.DataFrame({"a": [1], "b": [4], "c": [5]})]
    assert set(ChunkedFeatureGenerator._find_shared_cols(dfs=dfs)) == {"a", "b"}  # type: ignore


def test_remove_shared_cols_from_all_but_one_df():
    dfs = [pl.DataFrame({"a": [1], "b": [4]}), pl.DataFrame({"a": [1], "b": [4], "c": [5]})]
    shared_cols = ["a", "b"]
    dfs = ChunkedFeatureGenerator._remove_shared_cols_from_all_but_one_df(  # type: ignore
        dfs=dfs, shared_cols=shared_cols
    )
    assert dfs[0].columns == ["a", "b"]
    assert dfs[1].columns == ["c"]


def test_concatenation_assumptions(
    synth_predictor_1: pd.DataFrame,
    synth_predictor_2: pd.DataFrame,
    synth_prediction_times: pd.DataFrame,
    synth_project_info: ProjectInfo,
):
    """assert that timeseriesflattener outputs are sorted as expected"""

    predictor_specs = [
        PredictorSpec(
            timeseries_df=synth_predictor_1,
            feature_base_name="pred1",
            aggregation_fn=mean,
            fallback=np.nan,
            lookbehind_days=1,
        ),
        PredictorSpec(
            timeseries_df=synth_predictor_2,
            feature_base_name="pred2",
            aggregation_fn=mean,
            fallback=np.nan,
            lookbehind_days=1,
        ),
    ]

    # create a dataset with two predictors - chunking
    chunked_datasets = [
        create_flattened_dataset(
            project_info=synth_project_info,
            prediction_times_df=synth_prediction_times,
            feature_specs=[spec],
            add_birthdays=False,
        )
        for spec in predictor_specs
    ]
    # create a dataset with two predictors - no chunking
    full_dataset = pl.from_pandas(
        create_flattened_dataset(
            project_info=synth_project_info,
            prediction_times_df=synth_prediction_times,
            feature_specs=predictor_specs,  # type: ignore
            add_birthdays=False,
        )
    )

    chunked_datasets = [pl.from_pandas(df) for df in chunked_datasets]

    # merge chunk datasets and compare to full dataset
    merged_chunked_datasets = ChunkedFeatureGenerator.merge_feature_dfs(dfs=chunked_datasets)

    # assert that the merged chunked datasets are the same as the full dataset
    # ordering of columns might be a bit different due to imap so iterating over columns
    # and comparing them individually
    assert set(full_dataset.columns) == set(merged_chunked_datasets.columns)

    for col in full_dataset.columns:
        assert_series_equal(full_dataset[col], merged_chunked_datasets[col])
