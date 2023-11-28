import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.test_utils.str_to_df import str_to_pl_df


@BaselineRegistry.data.register("mock_regional_move_df")
def mock_regional_move_df() -> pl.LazyFrame:
    return str_to_pl_df(
        """dw_ek_borger,region,first_regional_move_timestamp
        1,vest,2020-03-01
        2,vest,2100-01-01""",
    ).lazy()


@BaselineRegistry.data.register("mock_split_id_series")
def mock_split_id_series() -> pl.Series:
    # TODO: make numpy array
    return pl.Series("dw_ek_borger", [1])


if __name__ == "__main__":
    x = mock_split_id_series()
