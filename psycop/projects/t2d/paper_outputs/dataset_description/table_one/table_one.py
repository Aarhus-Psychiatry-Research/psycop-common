# %%
#################
# Load the data #
#################
import polars as pl

from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

model_train_df = pl.concat(
    [
        get_best_eval_pipeline.inputs.get_flattened_split_as_lazyframe(split="train"),
        get_best_eval_pipeline.inputs.get_flattened_split_as_lazyframe(split="val"),
    ],
    how="vertical",
).with_columns(dataset=pl.format("0. train"))

test_dataset = get_best_eval_pipeline.inputs.get_flattened_split_as_lazyframe(
    split="test",
).with_columns(
    dataset=pl.format("test"),
)

flattened_combined = pl.concat([model_train_df, test_dataset], how="vertical").rename(
    {"prediction_time_uuid": "pred_time_uuid"},
)

# %%
pred_times_to_keep = (
    T2DCohortDefiner.get_filtered_prediction_times_bundle().prediction_times
)


# %%
pred_times_to_keep_with_uuid = pred_times_to_keep.lazy().with_columns(
    pred_time_uuid=pl.col("dw_ek_borger").cast(pl.Utf8)
    + "-"
    + pl.col("timestamp").dt.strftime(format="%Y-%m-%d-%H-%M-%S"),
)

# %%
eligible_prediction_times = flattened_combined.join(
    pred_times_to_keep_with_uuid,
    on="pred_time_uuid",
    how="inner",
)

# %%
####################
# Grouped by visit #
####################
import pandas as pd

from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    RowSpecification,
    get_psychiatric_diagnosis_row_specs,
)

visit_row_specs = [
    RowSpecification(
        source_col_name="pred_age_in_years",
        readable_name="Age",
        nonnormal=True,
    ),
    RowSpecification(
        source_col_name="age_grouped",
        readable_name="Age grouped",
        categorical=True,
    ),
    RowSpecification(
        source_col_name="pred_sex_female",
        readable_name="Female",
        categorical=True,
        values_to_display=[1],
    ),
    *get_psychiatric_diagnosis_row_specs(readable_col_names=model_train_df.columns),
    RowSpecification(
        source_col_name="eval_hba1c_within_9999_days_count_fallback_nan",
        readable_name="N HbA1c measurements prior to contact",
        nonnormal=True,
    ),
    RowSpecification(
        source_col_name="any_hba1c_before_visit",
        readable_name="HbA1c measured before visit",
        categorical=True,
        values_to_display=[1],
    ),
    RowSpecification(
        source_col_name="pred_hba1c_within_1095_days_mean_fallback_nan",
        readable_name="HbA1c within 3 years prior",
        nonnormal=False,
    ),
    RowSpecification(
        source_col_name="outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous",
        readable_name="Incident T2D within 3 years prior",
        categorical=True,
        values_to_display=[1],
    ),
]


age_bins = [18, *list(range(19, 90, 10))]

visit_flattened_df = (
    eligible_prediction_times.select(
        [
            r.source_col_name
            for r in visit_row_specs
            if r.source_col_name not in ["age_grouped", "any_hba1c_before_visit"]
        ]
        + ["dataset"],
    )
    .with_columns(
        any_hba1c_before_visit=pl.col(
            "eval_hba1c_within_9999_days_count_fallback_nan",
        ).is_not_null(),
    )
    .collect()
    .to_pandas()
)

visit_flattened_df["age_grouped"] = pd.Series(
    bin_continuous_data(visit_flattened_df["pred_age_in_years"], bins=age_bins)[0],
).astype(str)

# %%
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    create_table,
)

visit_table_one = create_table(
    row_specs=visit_row_specs,
    data=visit_flattened_df,
    groupby_col_name="dataset",
)

# %%
######################
# Grouped by patient #
######################
patient_row_specs = [
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
    RowSpecification(
        source_col_name="days_from_first_contact_to_outcome",
        readable_name="Days from first contact to outcome",
        nonnormal=False,
    ),
]

patient_df = (
    (
        eligible_prediction_times.groupby("dw_ek_borger")
        .agg(
            pred_sex_female=pl.col("pred_sex_female").first(),
            incident_t2d=pl.col(
                "outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous",
            ).max(),
            first_contact_timestamp=pl.col("timestamp").min(),
            outcome_timestamp=pl.col("timestamp_first_diabetes_lab_result").max(),
            dataset=pl.col("dataset").first(),
        )
        .select(
            [
                "pred_sex_female",
                "incident_t2d",
                "first_contact_timestamp",
                "outcome_timestamp",
                "dataset",
            ],
        )
        .with_columns(
            days_from_first_contact_to_outcome=(
                pl.col("outcome_timestamp") - pl.col("first_contact_timestamp")
            ),
        )
    )
    .collect()
    .to_pandas()
)

patient_df["days_from_first_contact_to_outcome"] = (
    patient_df["days_from_first_contact_to_outcome"]  # type: ignore
    / pd.Timedelta(days=1)
).astype(float)

patient_table_one = create_table(
    row_specs=patient_row_specs,
    data=patient_df,
    groupby_col_name="dataset",
)

# %%
############
# Combined #
############
combined = pd.concat([visit_table_one, patient_table_one])

get_best_eval_pipeline.paper_outputs.paths.tables.mkdir(parents=True, exist_ok=True)
combined.to_csv(
    get_best_eval_pipeline.paper_outputs.paths.tables / "descriptive_stats_table.csv",
)

# %%
# %load_ext autoreload
# %autoreload 2
