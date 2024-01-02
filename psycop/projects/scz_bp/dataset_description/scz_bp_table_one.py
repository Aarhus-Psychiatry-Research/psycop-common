"""Script to generate table 1 for the scz bp cohort

Two parts:
A) Outpatient contacts (prediction times)
- Number of outpatient contacts
- Sex
- Age
- Psychiatric diagnoses 2 years prior to contact (all main ICD-10 F categories)
- Diagnosis of BP within 3 years from the contact
- Diagnosis of SCZ within 3 years from the contact
- Number of admissions 2 years prior to the contact

B) Patients
- Number of patients
- Sex
- N SCZ
- N BP
- Days from first contact to outcome
- N admissions
- N prediction times (outpatient contacts)
"""


"""Part B

- get filtered IDs
"""
import re
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from confection import Config
from tableone import TableOne

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import admissions
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.common.model_training_v2.trainer.data.data_filters.geographical_split.make_geographical_split import (
    get_regional_split_df,
)
from psycop.common.model_training_v2.trainer.data.dataloaders import (
    ParquetVerticalConcatenator,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.add_time_from_first_visit import (
    time_of_first_contact_to_psychiatry,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.bp_diagnoses import (
    get_first_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.scz_diagnoses import (
    get_first_scz_diagnosis,
)
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    RowSpecification,
    create_table,
)


class SczBpTableOne:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def make_tables(self) -> list[pd.DataFrame]:
        pred_times = SczBpTableOne.get_filtered_prediction_times(self)
        section_a = self.make_section_A(pred_times=pred_times)
        section_b = self.make_section_B(pred_times)

        return [section_a, section_b]

    def make_section_A(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        """
        - Number of outpatient contacts
        - Sex
        - Age
        - Psychiatric diagnoses 2 years prior to contact (all main ICD-10 F categories)
        - Diagnosis of BP within 3 years from the contact
        - Diagnosis of SCZ within 3 years from the contact
        - Number of admissions 2 years prior to the contact
        """
        row_specifications = [
            RowSpecification(
                source_col_name="pred_sex_female_layer_1",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="age_grouped",
                readable_name="Age grouped",
                categorical=True,
            ),
            RowSpecification(
                source_col_name="pred_age_in_years", readable_name="Age", nonnormal=True
            ),
            RowSpecification(
                source_col_name="meta_bp_within_3_years_within_1095_days_maximum_fallback_0_dichotomous",
                readable_name="Incident BP within 3 years",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="meta_scz_within_3_years_within_1095_days_maximum_fallback_0_dichotomous",
                readable_name="Incident SCZ within 3 years",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="pred_admissions_layer_2_within_730_days_count_fallback_0",
                readable_name="N. admissions prior 2 years",
                nonnormal=True,
            ),
            *self.get_psychiatric_diagnosis_row_specs(col_names=pred_times.columns),
        ]

        age_bins = [18, *list(range(19, 90, 10))]

        pd_pred_times = (
            SczBpTableOne.add_split(pred_times)
            .select(
                [
                    r.source_col_name
                    for r in row_specifications
                    if r.source_col_name not in ["age_grouped"]
                ]
                + ["split"]
            )
            .to_pandas()
        )

        pd_pred_times["age_grouped"] = pd.Series(
            bin_continuous_data(pd_pred_times["pred_age_in_years"], bins=age_bins)[0],
        ).astype(str)

        return create_table(
            row_specs=row_specifications, data=pd_pred_times, groupby_col_name="split"
        )

    @staticmethod
    def get_psychiatric_diagnosis_row_specs(
        col_names: list[str]
    ) -> list[RowSpecification]:
        pattern = re.compile(r"pred_f\d_disorders")
        columns = sorted(
            [
                c
                for c in col_names
                if pattern.search(c) and "boolean" in c and "730" in c
            ],
        )

        readable_col_names = []

        for c in columns:
            icd_10_fx = re.findall(r"f\d+", c)[0]
            readable_col_name = f"{icd_10_fx.capitalize()} disorders prior 2 years"
            readable_col_names.append(readable_col_name)

        specs = []
        for i, _ in enumerate(readable_col_names):
            specs.append(
                RowSpecification(
                    source_col_name=columns[i],
                    readable_name=readable_col_names[i],
                    categorical=True,
                    nonnormal=False,
                    values_to_display=[1],
                ),
            )

        return specs

    def make_section_B(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        df = pl.DataFrame(pred_times["dw_ek_borger"].unique())

        fns = [
            SczBpTableOne.add_sex,
            SczBpTableOne.add_bipolar,
            SczBpTableOne.add_scz,
            SczBpTableOne.add_time_of_first_contact,
            SczBpTableOne.add_time_from_first_contact_to_bp_diagnosis,
            SczBpTableOne.add_time_from_first_contact_to_scz_diagnosis,
            partial(SczBpTableOne.add_n_prediction_times, pred_times=pred_times),
            SczBpTableOne.add_n_admissions,
            SczBpTableOne.add_split,
        ]
        for fn in fns:
            df = fn(df)

        df = df.select(pl.exclude("first_contact", "dw_ek_borger"))
        categorical = [
            "sex_female",
            "Incident BP",
            "Incident SCZ",
        ]
        non_normal = [
            "Days from first contact to BP diagnosis",
            "Days from first contact to SCZ diagnosis",
            "N. outpatient visits",
            "N. admissions",
        ]
        groupby = "split"

        return TableOne(
            data=df.to_pandas(),
            categorical=categorical,
            nonnormal=non_normal,
            groupby=groupby,
            missing=False,
        ).tableone

    def get_filtered_prediction_times(self) -> pl.DataFrame:
        # TODO: load from flattened df to get the actual filtered values
        data_paths = self.cfg["trainer"]["training_data"]["dataloader"]["paths"]

        data_dir = "/".join(data_paths[0].split("/")[:-1])

        all_splits = ParquetVerticalConcatenator(
            paths=[
                f"{data_dir}/train.parquet",
                f"{data_dir}/val.parquet",
                f"{data_dir}/test.parquet",
            ]
        ).load()
        preprocessing_pipeline = BaselineRegistry().resolve(
            {"pipe": self.cfg["trainer"]["preprocessing_pipeline"]}
        )

        preprocessed_all_splits: pl.DataFrame = pl.from_pandas(
            preprocessing_pipeline["pipe"].apply(all_splits)
        )

        # check if all_splits and preprocessed.. are differnet
        all_splits = all_splits.select(
            pl.col("^meta.*$"), "prediction_time_uuid"
        ).collect()
        preprocessed_all_splits = preprocessed_all_splits.join(
            all_splits, on="prediction_time_uuid", how="left"
        )

        return preprocessed_all_splits

    @staticmethod
    def add_sex(df: pl.DataFrame) -> pl.DataFrame:
        return df.join(
            pl.from_pandas(sex_female().rename({"sex_female": "Female"})),
            on="dw_ek_borger",
            how="left",
        )

    @staticmethod
    def add_bipolar(df: pl.DataFrame) -> pl.DataFrame:
        bp_df = pl.from_pandas(get_first_bp_diagnosis()).select(
            "dw_ek_borger",
            pl.col("value").alias("Incident BP"),
        )
        return df.join(bp_df, on="dw_ek_borger", how="left")

    @staticmethod
    def add_scz(df: pl.DataFrame) -> pl.DataFrame:
        scz_df = pl.from_pandas(get_first_scz_diagnosis()).select(
            "dw_ek_borger",
            pl.col("value").alias("Incident SCZ"),
        )
        return df.join(scz_df, on="dw_ek_borger", how="left")

    @staticmethod
    def add_time_of_first_contact(df: pl.DataFrame) -> pl.DataFrame:
        first_contact = time_of_first_contact_to_psychiatry()
        return df.join(first_contact, on="dw_ek_borger", how="left")

    @staticmethod
    def add_time_from_first_contact_to_bp_diagnosis(df: pl.DataFrame) -> pl.DataFrame:
        bp_df = pl.from_pandas(get_first_bp_diagnosis()).select(
            "dw_ek_borger",
            "timestamp",
        )
        return (
            df.join(bp_df, on="dw_ek_borger", how="left")
            .with_columns(
                ((pl.col("timestamp") - pl.col("first_contact")).dt.days()).alias(
                    "Days from first contact to BP diagnosis",
                ),
            )
            .select(pl.exclude("timestamp"))
        )

    @staticmethod
    def add_time_from_first_contact_to_scz_diagnosis(df: pl.DataFrame) -> pl.DataFrame:
        bp_df = pl.from_pandas(get_first_scz_diagnosis()).select(
            "dw_ek_borger",
            "timestamp",
        )
        return (
            df.join(bp_df, on="dw_ek_borger", how="left")
            .with_columns(
                ((pl.col("timestamp") - pl.col("first_contact")).dt.days()).alias(
                    "Days from first contact to SCZ diagnosis",
                ),
            )
            .select(pl.exclude("timestamp"))
        )

    @staticmethod
    def add_n_prediction_times(
        df: pl.DataFrame, pred_times: pl.DataFrame
    ) -> pl.DataFrame:
        n_prediction_times_pr_id = (
            pred_times.groupby("dw_ek_borger")
            .count()
            .rename({"count": "N. outpatient visits"})
        )
        return df.join(n_prediction_times_pr_id, on="dw_ek_borger", how="left")

    @staticmethod
    def add_n_admissions(df: pl.DataFrame) -> pl.DataFrame:
        admissions_pr_id = (
            pl.from_pandas(admissions())
            .groupby("dw_ek_borger")
            .count()
            .rename({"count": "N. admissions"})
        )
        return df.join(admissions_pr_id, on="dw_ek_borger", how="left")

    @staticmethod
    def add_split(df: pl.DataFrame) -> pl.DataFrame:
        split_df = (
            get_regional_split_df()
            .collect()
            .with_columns(
                pl.when(pl.col("region").is_in(["øst", "vest"]))
                .then("Train")
                .otherwise("test")
                .alias("split"),
            )
            .select("dw_ek_borger", "split")
        )
        return df.join(split_df, on="dw_ek_borger", how="left")


if __name__ == "__main__":
    populate_baseline_registry()
    cfg_path = OVARTACI_SHARED_DIR / "scz_bp" / "experiments" / "l3" / "config.cfg"
    save_dir = OVARTACI_SHARED_DIR / "scz_bp" / "dataset_description"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config().from_disk(cfg_path)
    resolved_cfg = BaselineRegistry().resolve(cfg)

    tables = SczBpTableOne(cfg=cfg).make_tables()
    tables[0].to_csv(save_dir / "table_1_prediction_times.csv")
    tables[1].to_csv(save_dir / "table_1_patients.csv")
