import pytest
from polars import ColumnNotFoundError

from psycop.common.feature_generation.sequences.timeseries_windower.types.prediction_time_dataframe import (
    PredictionTimeColumns,
    PredictiontimeDataframeBundle,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


class TestPredictionTimeBundle:
    @staticmethod
    def test_columns_do_not_exist():
        with pytest.raises(
            ColumnNotFoundError, match=r"entity_id"
        ):
            PredictiontimeDataframeBundle(
                _df=str_to_pl_df(
                    """e,timestamp
                    1,2021-01-01 00:00:00
                    1,2021-01-01 00:00:00
                    """
                ).lazy(),
                _cols=PredictionTimeColumns(
                    entity_id="entity_id", timestamp="timestamp"
                ),
            ).unpack()
