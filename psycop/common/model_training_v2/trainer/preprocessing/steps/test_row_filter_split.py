import polars as pl

from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByEntityID,
    RegionalFilter,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df

from ....config.baseline_registry import BaselineRegistry


def mock_data_for_id_filters() -> pl.LazyFrame:
    return str_to_pl_df(
        """dw_ek_borger,timestamp
            1, 2020-01-01
            1, 2020-02-01
            1, 2020-03-01
            2, 2020-01-01
            3, 2020-01-01
            """,
    ).lazy()


def test_regional_filter():
    input_df = mock_data_for_id_filters()
    regional_move_df = str_to_pl_df(
        """dw_ek_borger,region,first_regional_move_timestamp
        1,vest,2020-03-01
        2,vest,2100-01-01
        3,øst,2100-01-01""",
    ).lazy()

    regional_filter = RegionalFilter(
        splits_to_keep=["val"],
        regional_move_df=regional_move_df,
    )

    filtered_df = regional_filter.apply(input_df).collect()

    assert filtered_df.shape == (3, 2)


def test_id_filter():
    split_sequence = pl.Series("dw_ek_borger", [1])
    input_df = mock_data_for_id_filters()
    id_filter = FilterByEntityID(
        split_ids=split_sequence,
    )
    filtered_df = id_filter.apply(input_df).collect()

    assert filtered_df.shape == (3, 2)


@BaselineRegistry.data.register("mock_regional_move_df")
def mock_regional_move_df() -> pl.LazyFrame:
    return str_to_pl_df(
        """dw_ek_borger,region,first_regional_move_timestamp
        1,vest,2020-03-01
        2,vest,2100-01-01""",
    ).lazy()


@BaselineRegistry.data.register("mock_split_id_sequence")
def mock_split_id_sequence() -> list[int]:
    # converting to list as confection cannot resolve polars series
    return pl.Series("dw_ek_borger", [1]).to_list()
