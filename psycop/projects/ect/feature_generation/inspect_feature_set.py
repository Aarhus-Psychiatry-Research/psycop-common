import polars as pl

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR


import polars.selectors as cs
df = pl.read_parquet(OVARTACI_SHARED_DIR / "ect" / "feature_set" / "flattened_datasets" / "ect_feature_set" / "ect_feature_set.parquet")

df.columns

df.group_by("dw_ek_borger").agg(pl.col("outc_ect_value_within_0_to_60_days_max_fallback_0").max()).sum()

df.select(cs.contains("age"))