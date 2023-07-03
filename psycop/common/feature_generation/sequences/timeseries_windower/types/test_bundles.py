import pytest
from polars import ColumnNotFoundError

from psycop.common.feature_generation.sequences.timeseries_windower.types.event_dataframe import (
    EventColumnNames,
    EventDataframeBundle,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.prediction_time_dataframe import (
    PredictiontimeColumnNames,
    PredictiontimeDataframeBundle,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.sequence_dataframe import (
    SequenceColumnNames,
    SequenceDataframeBundle,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


class TestSequenceDataframeBundle:
    @staticmethod
    def test_should_error_if_column_missing():
        with pytest.raises(
            ColumnNotFoundError,
            match=r"entity_id",
        ):
            SequenceDataframeBundle(
                df=str_to_pl_df(
                    """e,timestamp
                    1,2021-01-01 00:00:00
                    1,2021-01-01 00:00:00
                    """,
                ).lazy(),
                cols=SequenceColumnNames(),
                validate_cols_exist_on_init=True,
            ).unpack()

    @staticmethod
    def test_should_be_frozen():
        with pytest.raises(  # noqa: PT012
            AttributeError,
            match=r"frozen",
        ):
            bundle = SequenceDataframeBundle(
                df=str_to_pl_df(
                    """e,timestamp
                    1,2021-01-01 00:00:00
                    """,
                ).lazy(),
                cols=SequenceColumnNames(),
                validate_cols_exist_on_init=False,
            )

            bundle._cols = SequenceColumnNames()  # type: ignore


class TestEventDataframeBundle:
    @staticmethod
    def test_should_error_if_column_missing():
        with pytest.raises(
            ColumnNotFoundError,
            match=r"entity_id",
        ):
            EventDataframeBundle(
                df=str_to_pl_df(
                    """e,timestamp
                    1,2021-01-01 00:00:00
                    1,2021-01-01 00:00:00
                    """,
                ).lazy(),
                cols=EventColumnNames(),
                validate_cols_exist_on_init=True,
            ).unpack()


class TestPredictionTimeBundle:
    @staticmethod
    def test_should_error_if_column_missing():
        with pytest.raises(
            ColumnNotFoundError,
            match=r"entity_id",
        ):
            PredictiontimeDataframeBundle(
                df=str_to_pl_df(
                    """e,timestamp
                    1,2021-01-01 00:00:00
                    1,2021-01-01 00:00:00
                    """,
                ).lazy(),
                cols=PredictiontimeColumnNames(
                    entity_id="entity_id",
                    timestamp="timestamp",
                ),
                validate_cols_exist_on_init=True,
            ).unpack()
