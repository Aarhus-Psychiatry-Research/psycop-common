from pathlib import Path

import pandas as pd
import polars as pl
from confection import Config

from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

from psycop.projects.restraint.model_evaluation.table_one_utils import get_psychiatric_diagnosis_row_specs, add_split
from psycop.projects.t2d.paper_outputs.dataset_description.table_one.table_one_lib import (
    RowSpecification,
    create_table,
)


class RestraintTableOne:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def prediction_times_table(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        row_specifications = [
            RowSpecification(
                source_col_name="outc_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive predictions",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_manual_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive predictions for manual restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive predictions for chemical restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of positive predictions for mechanical restraint",
                categorical=True,
                values_to_display=[1],
            )]

        
        pd_pred_times = (
            add_split(pred_times)
            .select(
                [
                    r.source_col_name
                    for r in row_specifications
                ]
                + ["split"]
            )
            .to_pandas()
        )
        
        return create_table(row_specs=row_specifications, data=pd_pred_times, groupby_col_name="split")
    
    def admissions_table(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        row_specifications = [
            RowSpecification(
                source_col_name="pred_sex_female_fallback_nan",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="pred_age_years_fallback_nan", readable_name="Age", nonnormal=True
            ),
            RowSpecification(
                source_col_name="age_grouped", readable_name="Age grouped", categorical=True
            ),
            *get_psychiatric_diagnosis_row_specs(col_names=pred_times.columns),
        ]

        age_bins: list[int] = [18, *list(range(20, 61, 10))]

        pd_pred_times = (
            add_split(pred_times)
            .select(
                [
                    r.source_col_name
                    for r in row_specifications
                    if r.source_col_name not in ["age_grouped"]
                ]
                + ["split", "dw_ek_borger", "timestamp_admission"]
            )
            .to_pandas()
        ).drop_duplicates(["dw_ek_borger", "timestamp_admission"])

        pd_pred_times["age_grouped"] = pd.Series(
            bin_continuous_data(pd_pred_times["pred_age_years_fallback_nan"], bins=age_bins)[0]
        ).astype(str)

        return create_table(
            row_specs=row_specifications, data=pd_pred_times, groupby_col_name="split"
        )
    

    def patients_table(self, pred_times: pl.DataFrame) -> pd.DataFrame:
        row_specifications = [
            RowSpecification(
                source_col_name="pred_sex_female_fallback_nan",
                readable_name="Female",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="pred_age_years_fallback_nan", readable_name="Age", nonnormal=True
            ),
            RowSpecification(
                source_col_name="age_grouped", readable_name="Age grouped", categorical=True
            ),
            RowSpecification(
                source_col_name="outc_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of incident restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_manual_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of incident manual restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of incident chemical restraint",
                categorical=True,
                values_to_display=[1],
            ),
            RowSpecification(
                source_col_name="outc_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous",
                readable_name="Number of incident mechanical restraint",
                categorical=True,
                values_to_display=[1],
            )]

        split_pred_times = add_split(pred_times)
        agg_pred_times = split_pred_times[["dw_ek_borger", "timestamp_admission", "pred_sex_female_fallback_nan", "pred_age_years_fallback_nan",
        "outc_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
        "outc_manual_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous", "split"]].group_by(["dw_ek_borger", "timestamp_admission", "split", "pred_sex_female_fallback_nan"]).agg(pl.col(["outc_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
        "outc_manual_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous"]).sum(), pl.col("pred_age_years_fallback_nan").mean())

        agg_pred_times = agg_pred_times.with_columns(pl.col("outc_all_restraint_within_2_days_boolean_fallback_0_dichotomous").replace(2, 1))
        agg_pred_times = agg_pred_times.with_columns(pl.col("outc_manual_restraint_within_2_days_boolean_fallback_0_dichotomous").replace(2, 1))
        agg_pred_times = agg_pred_times.with_columns(pl.col("outc_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous").replace(2, 1))
        agg_pred_times = agg_pred_times.with_columns(pl.col("outc_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous").replace(2, 1))

        age_bins: list[int] = [18, *list(range(20, 61, 10))]

        filtered_pred_times = (
            agg_pred_times
            .select(
                [
                    r.source_col_name
                    for r in row_specifications
                    if r.source_col_name not in ["age_grouped"]
                ]
                + ["split", "dw_ek_borger", "timestamp_admission"]
            )
        ) #.drop_duplicates(["dw_ek_borger", "timestamp_admission"])


        patient_frame = filtered_pred_times[["dw_ek_borger", "pred_sex_female_fallback_nan", "pred_age_years_fallback_nan",
        "outc_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
        "outc_manual_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous", "split"]].group_by(["dw_ek_borger", "split", "pred_sex_female_fallback_nan"]).agg(pl.col(["outc_all_restraint_within_2_days_boolean_fallback_0_dichotomous",
        "outc_manual_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_chemical_restraint_within_2_days_boolean_fallback_0_dichotomous", "outc_mechanical_restraint_within_2_days_boolean_fallback_0_dichotomous"]).sum(), pl.col("pred_age_years_fallback_nan").mean())

        pd_patient_frame = patient_frame.to_pandas()
        pd_patient_frame["age_grouped"] = pd.Series(
            bin_continuous_data(pd_patient_frame["pred_age_years_fallback_nan"], bins=age_bins)[0]
        ).astype(str)

        return create_table(
            row_specs=row_specifications, data=pd_patient_frame, groupby_col_name="split"
        )


if __name__ == "__main__":
    populate_baseline_registry()
    cfg_path = Path(__file__).parent / "eval_config.cfg"
    save_dir = Path(__file__).parent / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config().from_disk(cfg_path)
    resolved_cfg = BaselineRegistry().resolve(cfg)

    table_one = RestraintTableOne(cfg)
    pred_times = pl.read_parquet("E:/shared_resources/coercion/text/structured.parquet") # RestraintTableOne.get_filtered_prediction_times(self)
    
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
