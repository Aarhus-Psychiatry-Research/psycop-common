import datetime as dt
from collections.abc import Sequence

import polars as pl


def get_test_date_of_birth_df(patient_ids: Sequence[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dw_ek_borger": patient_ids,
            "timestamp": [dt.datetime(year=1990, month=1, day=1) for _ in patient_ids],
        }
    )
