# %%
#################
# Load the data #
#################
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,
)
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
)
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

run = MlflowClientWrapper().get_run(experiment_name="CVD", run_name="Layer 1")
cfg = run.get_config()

flattened_data = pl.read_parquet(cfg["trainer"]["training_data"]["paths"][0])
train_data = (
    RegionalFilter(["train", "val"])
    .apply(flattened_data.lazy())
    .with_columns(dataset=pl.lit("0. train"))
)
test_data = (
    RegionalFilter(["test"]).apply(flattened_data.lazy()).with_columns(dataset=pl.lit("test"))
)

flattened_combined = pl.concat([train_data, test_data], how="vertical").rename(
    {"prediction_time_uuid": "pred_time_uuid"}
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
    RowSpecification(source_col_name="pred_age_in_years", readable_name="Age", nonnormal=True),
    RowSpecification(source_col_name="age_grouped", readable_name="Age grouped", categorical=True),
    RowSpecification(
        source_col_name="pred_sex_female",
        readable_name="Female",
        categorical=True,
        values_to_display=[1],
    ),
    *get_psychiatric_diagnosis_row_specs(readable_col_names=train_data.columns),
]


age_bins = [18, *list(range(19, 90, 10))]

visit_flattened_df = (
    flattened_combined.select(
        [r.source_col_name for r in visit_row_specs if r.source_col_name not in ["age_grouped"]]
        + ["dataset"]
    )
    .collect()
    .to_pandas()
)

visit_flattened_df["age_grouped"] = pd.Series(
    bin_continuous_data(visit_flattened_df["pred_age_in_years"], bins=age_bins)[0]
).astype(str)

# %%
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    create_table,
)

visit_table_one = create_table(
    row_specs=visit_row_specs, data=visit_flattened_df, groupby_col_name="dataset"
)

# %%
######################
# Grouped by patient #
######################
outcome_col_name = cfg["trainer"]["outcome_col_name"]
patient_row_specs = [
    RowSpecification(
        source_col_name="pred_sex_female",
        readable_name="Female",
        categorical=True,
        values_to_display=[1],
    ),
    RowSpecification(
        source_col_name=outcome_col_name,
        readable_name="Incident CVD",
        categorical=True,
        values_to_display=[1],
    ),
]

patient_df = (
    (
        flattened_combined.groupby("dw_ek_borger")
        .agg(
            pl.col("pred_sex_female").first().alias("pred_sex_female"),
            pl.col(outcome_col_name).max().alias(outcome_col_name),
            pl.col("timestamp").min().alias("first_contact_timestamp"),
            pl.col("dataset").first().alias("dataset"),
        )
        .select(["pred_sex_female", outcome_col_name, "first_contact_timestamp", "dataset"])
    )
    .collect()
    .to_pandas()
)


patient_table_one = create_table(
    row_specs=patient_row_specs, data=patient_df, groupby_col_name="dataset"
)

# %%
############
# Combined #
############
combined = pd.concat([visit_table_one, patient_table_one])

# %%
# %load_ext autoreload
# %autoreload 2
