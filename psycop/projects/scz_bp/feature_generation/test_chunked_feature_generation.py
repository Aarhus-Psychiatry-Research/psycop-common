import pandas as pd
import polars as pl
import pytest
from timeseriesflattener import TimeseriesFlattener
from timeseriesflattener.aggregation_fns import mean
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.testing.load_synth_data import (
    load_synth_outcome,
    load_synth_prediction_times,
)

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.projects.scz_bp.feature_generation.chunked_feature_generation import (
    ChunkedFeatureGenerator,
)


@pytest.fixture()
def synth_outcome() -> pd.DataFrame:
    return load_synth_outcome()


@pytest.fixture()
def synth_project_info() -> ProjectInfo:
    pass


def test_find_shared_cols():
    dfs = [
        pl.DataFrame({"a": [1], "b": [4]}),
        pl.DataFrame({"a": [1], "b": [4], "c": [5]}),
    ]
    assert ChunkedFeatureGenerator._find_shared_cols(dfs=dfs) == ["a", "b"]  # type: ignore


def test_remove_shared_cols_from_all_but_one_df():
    dfs = [
        pl.DataFrame({"a": [1], "b": [4]}),
        pl.DataFrame({"a": [1], "b": [4], "c": [5]}),
    ]
    shared_cols = ["a", "b"]
    dfs = ChunkedFeatureGenerator._remove_shared_cols_from_all_but_one_df(  # type: ignore
        dfs=dfs, shared_cols=shared_cols
    )
    assert dfs[0].columns == ["a", "b"]
    assert dfs[1].columns == ["c"]


def test_concatenation():
    # test whether the concatenations match output without chunks
    dataset = TimeseriesFlattener(
        prediction_times_df=load_synth_prediction_times(),
        drop_pred_times_with_insufficient_look_distance=False,
    )
    predictor_spec = PredictorGroupSpec(
        named_dataframes=[
            NamedDataframe(df=synth_outcome, name="pred1"),
            NamedDataframe(df=synth_outcome, name="pred2"),
        ],
        lookbehind_days=[1, 2],
        aggregation_fns=[mean],
        fallback=[0],
        incident=False,
    )
    dataset.add_spec(predictor_spec)
