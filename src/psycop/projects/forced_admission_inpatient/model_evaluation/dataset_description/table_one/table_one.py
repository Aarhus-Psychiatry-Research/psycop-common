### LOAD DATA ###

import pandas as pd
import polars as pl
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.data_loader.data_loader import DataLoader
from psycop.projects.forced_admission_inpatient.model_evaluation.config import (
    TABLES_PATH,
)
from psycop.projects.forced_admission_inpatient.model_evaluation.dataset_description.table_one.table_one_utils import (
    RowSpecification,
    create_table,
    get_psychiatric_diagnosis_row_specs,
)


def load_feature_set() -> pl.DataFrame:
    """Loads the feature set with text features

    Returns:
        pd.DataFrame: Df with column denoting train or test
    """
    cfg, _ = setup(
        config_file_name="default_config.yaml",
        application_config_dir_relative_path="../../../../../../psycop-common/src/psycop/projects/forced_admission_inpatient/model_training/config/",
    )

    train_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="train")
    val_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="val")

    train_df["dataset"] = "train"
    val_df["dataset"] = "val"

    df = pd.concat([train_df, val_df])
    df = pl.DataFrame(df)

    return df


df = load_feature_set()

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
        source_col_name="outc_bool_within_182_days",
        readable_name="Forced admission within 6 months",
        categorical=True,
        values_to_display=[1],
    ),
    RowSpecification(
        source_col_name="days_from_first_contact_to_outcome",
        readable_name="Days from first contact to outcome",
        nonnormal=True,
    ),
]

patient_df = (
    df.groupby("dw_ek_borger")
    .agg(
        pred_sex_female=pl.col("pred_sex_female").first(),
        outc_bool_within_182_days=pl.col(
            "outc_bool_within_182_days",
        ).max(),
        first_contact_timestamp=pl.col("timestamp").min(),
        outcome_timestamp=pl.col("outc_timestamp").max(),
        dataset=pl.col("dataset").first(),
    )
    .select(
        [
            "pred_sex_female",
            "outc_bool_within_182_days",
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
).to_pandas()

patient_df["days_from_first_contact_to_outcome"] = patient_df[
    "days_from_first_contact_to_outcome"
] / pd.Timedelta(  # type: ignore
    days=1,
)

patient_table_one = create_table(
    row_specs=patient_row_specs,
    data=patient_df,
    groupby_col_name="dataset",
)


####################
# Grouped by visit #
####################
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
    *get_psychiatric_diagnosis_row_specs(readable_col_names=df.columns),
]


age_bins = [18, *list(range(19, 90, 10))]

visit_flattened_df = df.select(
    [
        r.source_col_name
        for r in visit_row_specs
        if r.source_col_name not in ["age_grouped"]
    ]
    + ["dataset"],
).to_pandas()

visit_flattened_df["age_grouped"] = pd.Series(
    bin_continuous_data(visit_flattened_df["pred_age_in_years"], bins=age_bins)[0],
).astype(str)


visit_table_one = create_table(
    row_specs=visit_row_specs,
    data=visit_flattened_df,
    groupby_col_name="dataset",
)

# Save the df to a csv file
visit_table_one.to_csv(TABLES_PATH / "visit_table_one.csv")
patient_table_one.to_csv(TABLES_PATH / "patient_table_one.csv")
