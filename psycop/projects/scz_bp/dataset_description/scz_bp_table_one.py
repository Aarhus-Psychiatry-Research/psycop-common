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

import pandas as pd
import polars as pl
from confection import Config
from tableone import TableOne

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_visits import admissions
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.dummy_logger import DummyLogger
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,  # type: ignore
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
        - Diagnosis of BP within 5 years from the contact
        - Diagnosis of SCZ within 5 years from the contact
        - Number of admissions 2 years prior to the contact
        """
        row_specifications = [
            RowSpecification(
                source_col_name="outc_first_scz_or_bp_within_0_to_1825_days_max_fallback_0",
                readable_name="Positive prediction time",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="meta_sex_female_fallback_0",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="age_grouped", readable_name="Age grouped", categorical=True
            ),
            RowSpecification(
                source_col_name="meta_age_years_fallback_nan", readable_name="Age", nonnormal=True
            ),
            RowSpecification(
                source_col_name="meta_bp_diagnosis_within_0_to_1825_days_max_fallback_0",
                readable_name="Incident BP within 5 years",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="meta_scz_diagnosis_within_0_to_1825_days_max_fallback_0",
                readable_name="Incident SCZ within 5 years",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="meta_n_admissions_within_0_to_730_days_count_fallback_0",
                readable_name="N. admissions prior 2 years",
                nonnormal=True,
            ),
            *self.get_psychiatric_diagnosis_row_specs(col_names=pred_times.columns),
        ]

        age_bins: list[int] = [15, 18, *list(range(20, 61, 10))]

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
            bin_continuous_data(pd_pred_times["meta_age_years_fallback_nan"], bins=age_bins)[0]
        ).astype(str)

        return create_table(
            row_specs=row_specifications, data=pd_pred_times, groupby_col_name="split"
        )

    @staticmethod
    def get_psychiatric_diagnosis_row_specs(col_names: list[str]) -> list[RowSpecification]:
        pattern = re.compile(r"pred_f\d_disorders")
        columns = sorted([c for c in col_names if pattern.search(c) and "730" in c])

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
                )
            )

        return specs

    def make_section_B(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        """By patients"""
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
        ]
        for fn in fns:
            df = fn(df)

        split_by_id = SczBpTableOne.get_split_by_id(pred_times=pred_times)
        df = split_by_id.join(df, on="dw_ek_borger", how="left")

        df = df.select(pl.exclude("first_contact", "dw_ek_borger"))
        categorical = ["sex_female", "Incident BP", "Incident SCZ"]
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
        data = BaselineRegistry.resolve({"data": self.cfg["trainer"]["training_data"]})[
            "data"
        ].load()

        preprocessing_pipeline = BaselineRegistry().resolve(
            {"pipe": self.cfg["trainer"]["preprocessing_pipeline"]}
        )["pipe"]
        preprocessing_pipeline._logger = DummyLogger()
        preprocessed_all_splits: pl.DataFrame = pl.from_pandas(preprocessing_pipeline.apply(data))

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
        bp_df = pl.from_pandas(get_first_bp_diagnosis()).select("dw_ek_borger", "timestamp")
        return (
            df.join(bp_df, on="dw_ek_borger", how="left")
            .with_columns(
                ((pl.col("timestamp") - pl.col("first_contact")).dt.days()).alias(
                    "Days from first contact to BP diagnosis"
                )
            )
            .select(pl.exclude("timestamp"))
        )

    @staticmethod
    def add_time_from_first_contact_to_scz_diagnosis(df: pl.DataFrame) -> pl.DataFrame:
        bp_df = pl.from_pandas(get_first_scz_diagnosis()).select("dw_ek_borger", "timestamp")
        return (
            df.join(bp_df, on="dw_ek_borger", how="left")
            .with_columns(
                ((pl.col("timestamp") - pl.col("first_contact")).dt.days()).alias(
                    "Days from first contact to SCZ diagnosis"
                )
            )
            .select(pl.exclude("timestamp"))
        )

    @staticmethod
    def add_n_prediction_times(df: pl.DataFrame, pred_times: pl.DataFrame) -> pl.DataFrame:
        n_prediction_times_pr_id = (
            pred_times.groupby("dw_ek_borger").count().rename({"count": "N. outpatient visits"})
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
    def get_split_by_id(pred_times: pl.DataFrame) -> pl.DataFrame:
        train_filter = RegionalFilter(splits_to_keep=["train", "val"])
        test_filter = RegionalFilter(splits_to_keep=["test"])

        train_data = (
            train_filter.apply(pred_times.lazy())
            .with_columns(pl.lit("train").alias("split"))
            .collect()
        )
        test_data = (
            test_filter.apply(pred_times.lazy())
            .with_columns(pl.lit("test").alias("split"))
            .collect()
        )

        df = pl.concat([train_data, test_data], how="vertical")
        any_id_in_multiple_splits = (
            df.group_by("dw_ek_borger")
            .agg(pl.col("split").n_unique().alias("n_splits_per_id"))
            .filter(pl.col("n_splits_per_id") > 1)
        )
        if not any_id_in_multiple_splits.is_empty():
            raise ValueError("Some patients are in multiple splits.")

        df = df.group_by("dw_ek_borger").agg(pl.col("split").first())
        return df

    @staticmethod
    def add_split(pred_times: pl.DataFrame) -> pl.DataFrame:
        train_filter = RegionalFilter(splits_to_keep=["train", "val"])
        test_filter = RegionalFilter(splits_to_keep=["test"])

        train_data = (
            train_filter.apply(pred_times.lazy())
            .with_columns(pl.lit("train").alias("split"))
            .collect()
        )
        test_data = (
            test_filter.apply(pred_times.lazy())
            .with_columns(pl.lit("test").alias("split"))
            .collect()
        )

        return pl.concat([train_data, test_data], how="vertical")


def descriptive_stats_by_lookahead(cfg: Config) -> pl.DataFrame:
    data = BaselineRegistry.resolve({"data": cfg["trainer"]["training_data"]})["data"].load()

    lookahead_distances = [365, 365 * 2, 365 * 3, 365 * 4, 365 * 5]
    cfg_copy = cfg.copy()

    dfs = []
    for lookahead_distance in lookahead_distances:
        cfg_copy["trainer"]["preprocessing_pipeline"]["*"]["lookahead_distance_filter"][
            "n_days"
        ] = lookahead_distance
        preprocessing_pipeline = BaselineRegistry().resolve(
            {"pipe": cfg_copy["trainer"]["preprocessing_pipeline"]}
        )
        preprocessing_pipeline = preprocessing_pipeline["pipe"]
        preprocessing_pipeline._logger = DummyLogger()
        d = {}

        preprocessed_data: pl.DataFrame = pl.from_pandas(preprocessing_pipeline.apply(data))

        positive_prediction_times = preprocessed_data.select(
            pl.col(f"^outc.*{lookahead_distance}.*$")
        ).sum()
        cols = positive_prediction_times.columns
        positive_prediction_times = positive_prediction_times.get_column(cols[0]).to_list()[0]

        n_positive_patients = (
            preprocessed_data.group_by("dw_ek_borger")
            .agg(pl.col(f"^outc.*{lookahead_distance}.*$").max())
            .sum()
        )
        n_positive_patients = n_positive_patients.get_column(cols[0]).to_list()[0]

        d["Lookahead time"] = lookahead_distance
        d["N prediction times"] = len(preprocessed_data)

        d["N positive prediction times"] = positive_prediction_times
        d["Proportion positive cases"] = positive_prediction_times / len(preprocessed_data)
        d["N unique patients"] = preprocessed_data.get_column("dw_ek_borger").n_unique()

        d["N unique positive patients"] = n_positive_patients
        dfs.append(pl.DataFrame(d))
    return pl.concat(dfs, how="vertical")


if __name__ == "__main__":
    populate_baseline_registry()
    cfg_path = Path(__file__).parent / "eval_config.cfg"
    save_dir = Path(__file__).parent / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config().from_disk(cfg_path)
    resolved_cfg = BaselineRegistry().resolve(cfg)

    # desc_stats = descriptive_stats_by_lookahead(cfg=cfg) # noqa: ERA001

    tables = SczBpTableOne(cfg=cfg).make_tables()  # noqa: ERA001
    with (save_dir / "table_1_prediction_times.html").open("w") as f:
        f.write(tables[0].to_html())
    with (save_dir / "table_1_patients.html").open("w") as f:
        f.write(tables[1].to_html())
    tables[0].to_csv(save_dir / "table_1_prediction_times.csv", sep=";")  # noqa: ERA001
    tables[1].to_csv(save_dir / "table_1_patients.csv", sep=";")  # noqa: ERA001
