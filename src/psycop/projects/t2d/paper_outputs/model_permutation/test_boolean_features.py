import polars as pl
from psycop.common.test_utils.str_to_df import str_to_df


def test_polars_to_boolean():
    pd_df = str_to_df(
        """id,timestamp,pred_test,pred_test2,outc_test
        1,2020-01-01 00:00:00,0,NaN,0
        1,2020-01-01 00:00:00,400,0,0
        1,2020-01-03 00:00:00,NaN,0,1
        """,
    )

    pl_df = pl.from_pandas(pd_df).lazy()

    pred_cols = [c for c in pl_df.columns if "pred_" in c]

    for col in pred_cols:
        pl_df = pl_df.with_columns(
            pl.when(pl.col(col).is_not_null()).then(1).otherwise(0).alias(col),
        )

    pl_df.collect()

    pass
