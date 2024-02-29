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
f_disorders = [f"f{i}_disorders" for i in range(9)]


from dataclasses import dataclass


@dataclass
class FeatureSpecification:
    name: str

    @property
    def source_column(self) -> str:
        return f"pred_{self.name}_within_1825_days_max_fallback_0"

    @property
    def aggregation(self) -> pl.Expr:
        return pl.col(self.source_column).max().alias(f"{self.name}")

    @property
    def row_specification(self) -> RowSpecification:
        return RowSpecification(
            source_col_name=self.name,
            readable_name=self.name.capitalize().replace("_", " "),
            categorical=True,
            values_to_display=[1],
        )


feature_specifications = [
    FeatureSpecification(n)
    for n in [
        *f_disorders,
        "top_10_weight_gaining_antipsychotics",
        "benzodiazepines",
        "benzodiazepine_related_sleeping_agents",
        "clozapine",
        "lamotrigine",
        "lithium",
        "pregabaline",
        "selected_nassa",
        "snri",
        "ssri",
        "tca",
        "valproate",
    ]
]

patient_df = (
    eligible_prediction_times.group_by("dw_ek_borger").agg(
        pl.col("pred_sex_female").first().alias("pred_sex_female"),
        pl.col("outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous")
        .max()
        .alias("incident_t2d"),
        pl.col("pred_age_in_years").min().alias("age"),
        *[f.aggregation for f in feature_specifications],
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
        *[f.row_specification for f in feature_specifications],
    ],
    data=patient_df,
    groupby_col_name="incident_t2d",
)
patient_table_one  # noqa: B018 # type: ignore

# %%
# %load_ext autoreload
# %autoreload 2

# %%
