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
import polars as pl
from tableone import TableOne

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import admissions
from psycop.common.model_training_v2.trainer.data.data_filters.geographical_split.make_geographical_split import (
    get_regional_split_df,
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


class SczBpTableOnePatients:
    def make_table(self) -> TableOne:
        pred_times = SczBpTableOnePatients.get_filtered_prediction_times()
        df = pl.DataFrame(pred_times["dw_ek_borger"].unique())

        fns = [
            SczBpTableOnePatients.add_sex,
            SczBpTableOnePatients.add_bipolar,
            SczBpTableOnePatients.add_scz,
            SczBpTableOnePatients.add_time_of_first_contact,
            SczBpTableOnePatients.add_time_from_first_contact_to_bp_diagnosis,
            SczBpTableOnePatients.add_time_from_first_contact_to_scz_diagnosis,
            SczBpTableOnePatients.add_n_prediction_times,
            SczBpTableOnePatients.add_n_admissions,
            SczBpTableOnePatients.add_split,
        ]
        for fn in fns:
            df = fn(df)

        df = df.select(pl.exclude("first_contact", "dw_ek_borger"))
        categorical = [
            "sex_female",
            "BP",
            "SCZ",
        ]
        non_normal = ["days_to_bp", "days_to_scz", "n_prediction_times", "n_admissions"]
        groupby = "split"

        return TableOne(
            data=df.to_pandas(),
            categorical=categorical,
            nonnormal=non_normal,
            groupby=groupby,
            missing=False,
        )

    @staticmethod
    def get_filtered_prediction_times() -> pl.DataFrame:
        # TODO: load from flattened df to get the actual filtered values
        return SczBpCohort.get_filtered_prediction_times_bundle().prediction_times

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
            "dw_ek_borger", pl.col("value").alias("Incident BP")
        )
        return df.join(bp_df, on="dw_ek_borger", how="left")

    @staticmethod
    def add_scz(df: pl.DataFrame) -> pl.DataFrame:
        scz_df = pl.from_pandas(get_first_scz_diagnosis()).select(
            "dw_ek_borger", pl.col("value").alias("Incident SCZ")
        )
        return df.join(scz_df, on="dw_ek_borger", how="left")

    @staticmethod
    def add_time_of_first_contact(df: pl.DataFrame) -> pl.DataFrame:
        first_contact = time_of_first_contact_to_psychiatry()
        return df.join(first_contact, on="dw_ek_borger", how="left")

    @staticmethod
    def add_time_from_first_contact_to_bp_diagnosis(df: pl.DataFrame) -> pl.DataFrame:
        bp_df = pl.from_pandas(get_first_bp_diagnosis()).select(
            "dw_ek_borger", "timestamp"
        )
        return (
            df.join(bp_df, on="dw_ek_borger", how="left")
            .with_columns(
                (
                    ((pl.col("timestamp") - pl.col("first_contact")).dt.days()).alias(
                        "Days from first contact to BP diagnosis"
                    )
                )
            )
            .select(pl.exclude("timestamp"))
        )

    @staticmethod
    def add_time_from_first_contact_to_scz_diagnosis(df: pl.DataFrame) -> pl.DataFrame:
        bp_df = pl.from_pandas(get_first_scz_diagnosis()).select(
            "dw_ek_borger", "timestamp"
        )
        return (
            df.join(bp_df, on="dw_ek_borger", how="left")
            .with_columns(
                (
                    ((pl.col("timestamp") - pl.col("first_contact")).dt.days()).alias(
                        "Days from first contact to SCZ diagnosis"
                    )
                )
            )
            .select(pl.exclude("timestamp"))
        )

    @staticmethod
    def add_n_prediction_times(df: pl.DataFrame) -> pl.DataFrame:
        n_prediction_times_pr_id = (
            SczBpTableOnePatients.get_filtered_prediction_times()
            .groupby("dw_ek_borger")
            .count()
            .rename({"count": "N. outpatient visit"})
        )
        return df.join(n_prediction_times_pr_id, on="dw_ek_borger", how="left")

    @staticmethod
    def add_n_admissions(df: pl.DataFrame) -> pl.DataFrame:
        admissions_pr_id = (
            pl.from_pandas(admissions())
            .groupby("dw_ek_borger")
            .count()
            .rename({"count": "N. Admissions"})
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
                .otherwise("Test")
                .alias("split")
            )
            .select("dw_ek_borger", "split")
        )
        return df.join(split_df, on="dw_ek_borger", how="left")


if __name__ == "__main__":
    tab_b = SczBpTableOnePatients().make_table()
