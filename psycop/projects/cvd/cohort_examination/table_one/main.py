# %%
#################
# Load the data #
#################
from collections.abc import Sequence

import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,
)
from psycop.projects.cvd.cohort_examination.label_by_outcome_type import label_by_outcome_type
from psycop.projects.cvd.cohort_examination.table_one.facade import table_one
from psycop.projects.cvd.cohort_examination.table_one.model import table_one_model
from psycop.projects.cvd.cohort_examination.table_one.view import RowSpecification
from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_cvd_indicator,
)

run = MlflowClientWrapper().get_run(experiment_name="CVD", run_name="CVD layer 1, base")
model = table_one_model(run)

# %%
####################
# Grouped by visit #
####################
import pandas as pd

from psycop.projects.cvd.cohort_examination.table_one.view import (
    get_psychiatric_diagnosis_row_specs,
)

visit_row_specs = [
    RowSpecification(source_col_name="pred_age_in_years", readable_name="Age", nonnormal=True),
    RowSpecification(source_col_name="age_grouped", readable_name="Age grouped", categorical=True),
    RowSpecification(
        source_col_name="pred__sex_female_fallback_0",
        readable_name="Female",
        categorical=True,
        values_to_display=[1],
    ),
    *get_psychiatric_diagnosis_row_specs(readable_col_names=train_data.columns),
]
visit_row_specs  # noqa: B018


# %%
from psycop.common.model_evaluation.utils import bin_continuous_data

with_age_col = flattened_combined.with_columns(
    (pl.col("pred_age_days_fallback_0") / 365.25).alias("pred_age_in_years")
)

visit_flattened_df = (
    with_age_col.select(
        [r.source_col_name for r in visit_row_specs if r.source_col_name not in ["age_grouped"]]
        + ["dataset"]
    )
    .collect()
    .to_pandas()
)

visit_flattened_df["age_grouped"] = pd.Series(
    bin_continuous_data(
        visit_flattened_df["pred_age_in_years"], bins=[18, *list(range(19, 90, 10))]
    )[0]
).astype(str)

# %%
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    create_table,
)

visit_table_one = create_table(
    row_specs=visit_row_specs,  # type: ignore
    data=visit_flattened_df.fillna(0),
    groupby_col_name="dataset",
)
visit_table_one  # noqa: B018

# %%
######################
# Grouped by patient #
######################
outcome_col_name = cfg["trainer"]["outcome_col_name"]
sex_col_name = "pred__sex_female_fallback_0"
patient_row_specs = [
    RowSpecification(
        source_col_name=sex_col_name,
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
            pl.col(sex_col_name).first().alias(sex_col_name),
            pl.col(outcome_col_name).max().alias(outcome_col_name),
            pl.col("timestamp").min().alias("first_contact_timestamp"),
            pl.col("dataset").first().alias("dataset"),
        )
        .select([sex_col_name, outcome_col_name, "first_contact_timestamp", "dataset"])
    )
    .collect()
    .to_pandas()
)


patient_table_one = create_table(
    row_specs=patient_row_specs,  # type: ignore
    data=patient_df.fillna(0),
    groupby_col_name="dataset",
)

# %%
#################
# Outcome table #
#################
outcome_data = pl.from_pandas(get_first_cvd_indicator())
outcome_data_filtered = outcome_data.filter(
    pl.col("dw_ek_borger").is_in(flattened_combined.collect().get_column("dw_ek_borger"))
)

# %%
train_data = (
    RegionalFilter(["train", "val"])
    .apply(outcome_data_filtered.lazy())
    .with_columns(dataset=pl.lit("0. train"))
)
test_data = (
    RegionalFilter(["test"])
    .apply(outcome_data_filtered.lazy())
    .with_columns(dataset=pl.lit("test"))
)

outcome_combined = pl.concat([train_data, test_data], how="vertical").collect()

# %%
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    create_table,
)

outcome_table = create_table(
    row_specs=[RowSpecification("outcome_type", "CVD by type", True)],  # type: ignore
    data=label_by_outcome_type(outcome_combined, group_col="cause").to_pandas(),
    groupby_col_name="dataset",
)

# %%
