from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from psycop.common.feature_generation.data_checks.flattened.feature_describer_tsflattener_v2 import (
    ParsedPredictorColumn,
    generate_feature_description_df,
    parse_predictor_column_name,
    tsflattener_v2_column_is_static,
)


@dataclass
class PredictorColumnExample:
    col_name: str
    feature_name: str
    fallback: str
    time_interval_start: str
    time_interval_end: str
    resolve_multiple_strategy: str


@pytest.mark.parametrize(
    "predictor_column_example",
    [
        ParsedPredictorColumn(
            col_name="pred_value_1_within_0_to_730_days_mean_fallback_nan",
            feature_name="value_1",
            fallback="nan",
            time_interval_start="0",
            time_interval_end="730",
            time_interval_format="days",
            resolve_multiple_strategy="mean",
            is_static=False,
        ),
        ParsedPredictorColumn(
            col_name="pred_female_fallback_nan",
            feature_name="female",
            fallback="nan",
            time_interval_start="N/A",
            time_interval_end="N/A",
            time_interval_format="N/A",
            resolve_multiple_strategy="N/A",
            is_static=True,
        ),
    ],
)
def test_parse_predictor_column_name(predictor_column_example: PredictorColumnExample):
    parsed_col = parse_predictor_column_name(predictor_column_example.col_name)
    assert parsed_col == predictor_column_example


def test_tsflattener_v2_is_static():
    assert (
        tsflattener_v2_column_is_static("pred_value_1_within_0_to_730_days_mean_fallback_nan")
        is False
    )
    assert tsflattener_v2_column_is_static("pred_female_fallback_nan") is True


def test_generate_feature_description_df():
    df = pl.DataFrame(
        {
            "pred_female_fallback_nan": [0, 1, np.nan],
            "outc_first_scz_or_bp_within_0_to_730_days_max_fallback_0": [0, 0, 1],
            "pred_value_1_within_0_to_730_days_mean_fallback_nan": [1, 2, 3],
        }
    )

    expected = pl.DataFrame(
        {
            "Feature name": ["female", "first_scz_or_bp", "value_1"],
            "Lookbehind period": ["N/A", "0 to 730 days", "0 to 730 days"],
            "Resolve multiple": ["N/A", "max", "mean"],
            "Fallback strategy": ["nan", "0", "nan"],
            "Static": [True, False, False],
            "Proportion missing": [0.33, 0.0, 0.0],
            "Mean": [0.5, 0.33, 2.0],
            "N. unique": [3, 2, 3],
            "Proportion using fallback": [0.33, 0.67, 0.0],
        }
    )

    feature_description = generate_feature_description_df(
        df, column_name_parser=parse_predictor_column_name
    )
    assert_frame_equal(feature_description, expected)
