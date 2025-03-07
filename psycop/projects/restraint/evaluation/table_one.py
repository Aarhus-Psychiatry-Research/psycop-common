from pathlib import Path
from typing import Any, Dict

import pandas as pd
import polars as pl
from confection import Config

from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.config_utils import resolve_and_fill_config
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger
from psycop.projects.restraint.evaluation.utils import add_stratified_split, get_psychiatric_diagnosis_row_specs, add_admission_timestamps
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    RowSpecification,
    create_table,
)


class RestraintTableOne:
    def get_filtered_prediction_times(self, cfg: Dict[str, Any]):
        data = cfg["trainer"].training_data.load()

        preprocessing_pipeline = cfg["trainer"].preprocessing_pipeline
        preprocessed_all_splits: pl.DataFrame = pl.from_pandas(preprocessing_pipeline.apply(data))

        return preprocessed_all_splits

    def prediction_times_table(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        row_specifications = [
            RowSpecification(
                source_col_name="pred_sex_female",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive labels",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_manual_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive labels for manual restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive labels for chemical restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive labels for mechanical restraint",
                categorical=True,
                values_to_display=[1],
            ),
        ]

        pd_pred_times = (
            add_stratified_split(pred_times)
            .select([r.source_col_name for r in row_specifications] + ["split"])
            .to_pandas()
        )

        return create_table(
            row_specs=row_specifications, data=pd_pred_times, groupby_col_name="split"
        )

    def admissions_table(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        row_specifications = [
            RowSpecification(
                source_col_name="pred_sex_female",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="pred_age_in_years", readable_name="Age", nonnormal=True
            ),
            RowSpecification(
                source_col_name="age_grouped", readable_name="Age grouped", categorical=True
            ),
            *get_psychiatric_diagnosis_row_specs(col_names=pred_times.columns),
            RowSpecification(
                source_col_name="outc_outcome_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of outcome events",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_manual_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of outcome events for manual restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of outcome events for chemical restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of outcome events for mechanical restraint",
                categorical=True,
                values_to_display=[1],
            ),
        ]

        age_bins: list[int] = [18, *list(range(20, 61, 10))]
        pred_times = add_admission_timestamps(pred_times)
        pd_pred_times = add_stratified_split(pred_times).select(
                [
                    r.source_col_name
                    for r in row_specifications
                    if r.source_col_name not in ["age_grouped"]
                ]
                + ["split", "dw_ek_borger", "timestamp_admission"]
            ).sort(["dw_ek_borger", "timestamp_admission"])

        pd_pred_times = pd_pred_times.group_by(["dw_ek_borger", "timestamp_admission", "split", "pred_sex_female", 'pred_f0_disorders_within_730_days_boolean_fallback_0', 'pred_f1_disorders_within_730_days_boolean_fallback_0', 'pred_f2_disorders_within_730_days_boolean_fallback_0', 'pred_f3_disorders_within_730_days_boolean_fallback_0', 'pred_f4_disorders_within_730_days_boolean_fallback_0', 'pred_f5_disorders_within_730_days_boolean_fallback_0', 'pred_f6_disorders_within_730_days_boolean_fallback_0', 'pred_f7_disorders_within_730_days_boolean_fallback_0', 'pred_f8_disorders_within_730_days_boolean_fallback_0', 'pred_f9_disorders_within_730_days_boolean_fallback_0']).agg(
                (pl.max("outc_outcome_all_restraint_within_2_days_boolean_fallback_0_dichotomous"), 
                        pl.max("outc_outcome_manual_restraint_within_2_days_boolean_fallback_0_dichotomous"),
                        pl.max("outc_outcome_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous"),
                        pl.max("outc_outcome_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous"),
                        pl.first("pred_age_in_years")
                )).sort(["dw_ek_borger", "timestamp_admission"])

        binned_age = bin_continuous_data(pd.Series(pd_pred_times["pred_age_in_years"]), bins=age_bins)[0].astype(str) # type: ignore
        pd_pred_times = pl.concat([pd_pred_times, pl.DataFrame({"age_grouped": binned_age})], how="horizontal").to_pandas()

        return create_table(
            row_specs=row_specifications, data=pd_pred_times, groupby_col_name="split"
        )

    def patients_table(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        row_specifications = [
            RowSpecification(
                source_col_name="pred_sex_female",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of patients with at least one outcome event",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_manual_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of patients with manual restraint event",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of patients with chemical restraint event",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_outcome_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of patients with mechanical restraint event",
                categorical=True,
                values_to_display=[1],
            ),
        ]

        pred_times = add_admission_timestamps(pred_times)
        split_pred_times = add_stratified_split(pred_times)
        agg_pred_times = (
            split_pred_times[
                [
                    "dw_ek_borger",
                    "pred_sex_female",
                    "outc_outcome_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
                    "outc_outcome_manual_restraint_within_2_days_boolean_fallback_0_dichotomous",
                    "outc_outcome_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                    "outc_outcome_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                    "split",
                ]
            ]
            .group_by(
                ["dw_ek_borger", "split", "pred_sex_female"]
            )
            .agg(
                pl.col(
                    [
                        "outc_outcome_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
                        "outc_outcome_manual_restraint_within_2_days_boolean_fallback_0_dichotomous",
                        "outc_outcome_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                        "outc_outcome_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                    ]
                ).max(),
            )
        )

        agg_pred_times = agg_pred_times.with_columns(
            pl.col("outc_outcome_all_restraint_within_2_days_boolean_fallback_0_dichotomous").replace(2, 1)
        )
        agg_pred_times = agg_pred_times.with_columns(
            pl.col("outc_outcome_manual_restraint_within_2_days_boolean_fallback_0_dichotomous").replace(
                2, 1
            )
        )
        agg_pred_times = agg_pred_times.with_columns(
            pl.col("outc_outcome_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous").replace(
                2, 1
            )
        )
        agg_pred_times = agg_pred_times.with_columns(
            pl.col(
                "outc_outcome_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous"
            ).replace(2, 1)
        )

        pd_patient_frame = agg_pred_times.to_pandas()

        return create_table(
            row_specs=row_specifications, data=pd_patient_frame, groupby_col_name="split"
        )


if __name__ == "__main__":
    populate_baseline_registry()
    cfg_path = Path(__file__).parent / "table_one.cfg"
    save_dir = Path(__file__).parent / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = resolve_and_fill_config(cfg_path, fill_cfg_with_defaults=True)
    resolved_cfg["trainer"].preprocessing_pipeline._logger = TerminalLogger()

    table_one = RestraintTableOne()
    pred_times = table_one.get_filtered_prediction_times(resolved_cfg)

    prediction_times_table = table_one.prediction_times_table(pred_times)
    admissions_table = table_one.admissions_table(pred_times)
    patients_table = table_one.patients_table(pred_times)

    with (save_dir / "patients_table.html").open("w") as f:
        f.write(patients_table.to_html())
    with (save_dir / "admissions_table.html").open("w") as f:
        f.write(admissions_table.to_html())
    with (save_dir / "prediction_times_table.html").open("w") as f:
        f.write(prediction_times_table.to_html())
    patients_table.to_csv(save_dir / "patients_table.csv", sep=";")
    admissions_table.to_csv(save_dir / "admissions_table.csv", sep=";")
    prediction_times_table.to_csv(save_dir / "prediction_times_table.csv", sep=";")
