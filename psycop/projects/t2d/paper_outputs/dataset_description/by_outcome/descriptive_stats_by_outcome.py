# %%
#################
# Load the data #
#################
import polars as pl

from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    RowSpecification,
    create_table,
)
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

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
).collect()

# %%
######################
# Grouped by patient #
######################
patient_df = (
    eligible_prediction_times.group_by("dw_ek_borger").agg(
        pl.col("pred_sex_female").first().alias("pred_sex_female"),
        pl.col("outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous")
        .max()
        .alias("incident_t2d"),
        pl.col("pred_age_in_years").min().alias("age"),
        pl.col("pred_f0_disorders_within_1825_days_max_fallback_0").max().alias("f0_disorders"),
        pl.col("pred_f1_disorders_within_1825_days_max_fallback_0").max().alias("f1_disorders"),
        pl.col("pred_f2_disorders_within_1825_days_max_fallback_0").max().alias("f2_disorders"),
        pl.col("pred_f3_disorders_within_1825_days_max_fallback_0").max().alias("f3_disorders"),
        pl.col("pred_f4_disorders_within_1825_days_max_fallback_0").max().alias("f4_disorders"),
        pl.col("pred_f5_disorders_within_1825_days_max_fallback_0").max().alias("f5_disorders"),
        pl.col("pred_f6_disorders_within_1825_days_max_fallback_0").max().alias("f6_disorders"),
        pl.col("pred_f7_disorders_within_1825_days_max_fallback_0").max().alias("f7_disorders"),
        pl.col("pred_f8_disorders_within_1825_days_max_fallback_0").max().alias("f8_disorders"),
    )
).to_pandas()
patient_df  # noqa: B018 # type: ignore

# %%
patient_table_one = create_table(
    row_specs=[
        RowSpecification(
            source_col_name="pred_sex_female",
            readable_name="Female",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(source_col_name="age", readable_name="Age", nonnormal=True),
        RowSpecification(
            source_col_name="f0_disorders",
            readable_name="F0",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f1_disorders",
            readable_name="F1",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f2_disorders",
            readable_name="F2",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f3_disorders",
            readable_name="F3",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f4_disorders",
            readable_name="F4",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f5_disorders",
            readable_name="F5",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f6_disorders",
            readable_name="F6",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f7_disorders",
            readable_name="F7",
            categorical=True,
            values_to_display=[1],
        ),
        RowSpecification(
            source_col_name="f8_disorders",
            readable_name="F8",
            categorical=True,
            values_to_display=[1],
        ),
    ],
    data=patient_df,
    groupby_col_name="incident_t2d",
)
patient_table_one  # noqa: B018 # type: ignore

# %%
# %load_ext autoreload
# %autoreload 2

# %%
