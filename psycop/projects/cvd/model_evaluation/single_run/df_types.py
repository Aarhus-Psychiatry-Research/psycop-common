from typing import NewType

import polars as pl

PredTimestampDF = NewType("PredTimestampDF", pl.DataFrame)
# Must contain columns "timestamp" and "dw_ek_borger"

OutcomeTimestampDF = NewType("OutcomeTimestampDF", pl.DataFrame)
# Must contain columns "timestamp" and "dw_ek_borger"
