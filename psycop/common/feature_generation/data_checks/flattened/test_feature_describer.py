from dataclasses import dataclass

import pytest

from psycop.common.feature_generation.data_checks.flattened.feature_describer_tsflattener_v2 import (
    ParsedPredictorColumn,
    parse_predictor_column_name,
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
            resolve_multiple_strategy="mean",
            is_static=False,
        ),
        ParsedPredictorColumn(
            col_name="pred_female_fallback_nan",
            feature_name="female",
            fallback="nan",
            time_interval_start="N/A",
            time_interval_end="N/A",
            resolve_multiple_strategy="N/A",
            is_static=True,
        ),
    ],
)
def test_parse_predictor_column_name(predictor_column_example: PredictorColumnExample):
    parsed_col = parse_predictor_column_name(predictor_column_example.col_name)
