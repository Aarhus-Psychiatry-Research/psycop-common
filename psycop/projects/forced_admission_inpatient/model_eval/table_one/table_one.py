# %%
#################
# Load the data #
#################
import polars as pl

from psycop.projects.forced_admission_inpatient.feature_generation.modules.loaders.load_forced_admissions_dfs_with_prediction_times_and_outcome import (
    forced_admissions_inpatient,
)
from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)

model_train_df = pl.concat(
    [
        get_best_eval_pipeline.inputs.get_flattened_split_as_lazyframe(split="train"),
    ],
    how="vertical",
).with_columns(dataset=pl.format("0. train"))


val_dataset = get_best_eval_pipeline.inputs.get_flattened_split_as_lazyframe(
    split="val",
).with_columns(
    dataset=pl.format("val"),
)

flattened_combined = pl.concat([model_train_df, val_dataset], how="vertical").rename(
    {"prediction_time_uuid": "pred_time_uuid"},
)

# %%
pred_times_to_keep = pl.from_pandas(
    forced_admissions_inpatient(
        timestamps_only=True,
    ),
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
from psycop.projects.forced_admission_inpatient.model_eval.table_one.table_one_lib import (
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
]

age_bins = [18, *list(range(19, 90, 10))]

visit_flattened_df = (
    eligible_prediction_times.select(
        [
            r.source_col_name
            for r in visit_row_specs
            if r.source_col_name not in ["age_grouped"]
        ]
        + ["dataset"],
    )
    .collect()
    .to_pandas()
)

visit_flattened_df["age_grouped"] = pd.Series(
    bin_continuous_data(visit_flattened_df["pred_age_in_years"], bins=age_bins)[0],
).astype(str)

# %%
from psycop.projects.forced_admission_inpatient.model_eval.table_one.table_one_lib import (
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
]

patient_df = (
    (
        eligible_prediction_times.groupby("dw_ek_borger")
        .agg(
            pred_sex_female=pl.col("pred_sex_female").first(),
            dataset=pl.col("dataset").first(),
        )
        .select(
            [
                "pred_sex_female",
                "dataset",
            ],
        )
    )
    .collect()
    .to_pandas()
)

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
