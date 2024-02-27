# %%
#################
# Load the data #
#################
import polars as pl

from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

from ..table_one.table_one_lib import RowSpecification, create_table

base_df = pl.concat(
    [
        get_best_eval_pipeline().inputs.get_flattened_split_as_lazyframe(split="train"),
        get_best_eval_pipeline().inputs.get_flattened_split_as_lazyframe(split="val"),
        get_best_eval_pipeline().inputs.get_flattened_split_as_lazyframe(split="test"),
    ],
    how="vertical",
).rename({"prediction_time_uuid": "pred_time_uuid"})

# %%
pred_times_to_keep = T2DCohortDefiner.get_filtered_prediction_times_bundle().prediction_times

# %%
pred_times_to_keep_with_uuid = pred_times_to_keep.frame.lazy().with_columns(
    pred_time_uuid=pl.col("dw_ek_borger").cast(pl.Utf8)
    + "-"
    + pl.col("timestamp").dt.strftime(format="%Y-%m-%d-%H-%M-%S")
)

# %%
eligible_prediction_times = base_df.join(
    pred_times_to_keep_with_uuid, on="pred_time_uuid", how="inner"
)

# %%
######################
# Grouped by patient #
######################
patient_df = (
    (
        eligible_prediction_times.groupby("dw_ek_borger")
        .agg(
            pl.col("pred_sex_female").first().alias("pred_sex_female"),
            pl.col("outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous")
            .max()
            .alias("incident_t2d"),
            pl.col("pred_f0_disorders").max().alias("f0_disorders"),
            pl.col("pred_f1_disorders").max().alias("f1_disorders"),
            pl.col("pred_f2_disorders").max().alias("f2_disorders"),
            pl.col("pred_f3_disorders").max().alias("f3_disorders"),
            pl.col("pred_f4_disorders").max().alias("f4_disorders"),
            pl.col("pred_f5_disorders").max().alias("f5_disorders"),
            pl.col("pred_f6_disorders").max().alias("f6_disorders"),
            pl.col("pred_f7_disorders").max().alias("f7_disorders"),
            pl.col("pred_f8_disorders").max().alias("f8_disorders"),
        )
        .select(
            [
                "pred_sex_female",
                "incident_t2d",
                "first_contact_timestamp",
                "outcome_timestamp",
                "dataset",
            ]
        )
    )
    .collect()
    .to_pandas()
)

# %%
patient_table_one = create_table(
    row_specs=[
        RowSpecification(
            source_col_name="pred_sex_female",
            readable_name="Female",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="incident_t2d",
            readable_name="Incident T2D",
            categorical=True,
            values_to_display=[1],
        ),
    ],
    data=patient_df,
    groupby_col_name="dataset",
)

# %%
# %load_ext autoreload
# %autoreload 2
