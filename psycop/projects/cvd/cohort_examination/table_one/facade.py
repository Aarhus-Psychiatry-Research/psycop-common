from collections.abc import Sequence
from pathlib import Path

import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.projects.cvd.cohort_examination.table_one.model import table_one_model
from psycop.projects.cvd.cohort_examination.table_one.view import (
    ColumnOverride,
    RowCategory,
    table_one_view,
)

cvd_overrides = [
    ColumnOverride(
        "categorical",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.diagnoses,
    ),
    ColumnOverride(
        "lung",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.diagnoses,
    ),
    ColumnOverride(
        "antipsychotics",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.medications,
    ),
    ColumnOverride(
        "atrial",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.diagnoses,
    ),
    ColumnOverride(
        "antihypertensives",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.medications,
    ),
    ColumnOverride(
        "ldl",
        categorical=False,
        override_name="LDL",
        category=RowCategory.lab_results,
        values_to_display=None,
    ),
    ColumnOverride(
        "systolic",
        categorical=False,
        override_name=None,
        category=RowCategory.lab_results,
        values_to_display=None,
    ),
    ColumnOverride(
        "smoking_categorical",
        categorical=True,
        values_to_display=None,
        override_name=None,
        category=RowCategory.demographics,
    ),
    ColumnOverride(
        "smoking_continuous",
        categorical=False,
        override_name=None,
        category=RowCategory.demographics,
        values_to_display=None,
    ),
    ColumnOverride(
        "hba1c",
        categorical=False,
        override_name="HbA1c",
        category=RowCategory.lab_results,
        values_to_display=None,
    ),
    ColumnOverride(
        "hdl",
        categorical=False,
        override_name="HDL",
        category=RowCategory.lab_results,
        values_to_display=None,
    ),
    ColumnOverride(
        "type_1_diabetes",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.diagnoses,
    ),
    ColumnOverride(
        "type_2_diabetes",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.diagnoses,
    ),
    ColumnOverride(
        "weight_in_kg",
        categorical=False,
        override_name="Weight (kg)",
        category=RowCategory.demographics,
        values_to_display=None,
    ),
    ColumnOverride(
        "height",
        categorical=False,
        override_name="Height (cm)",
        category=RowCategory.demographics,
        values_to_display=None,
    ),
    ColumnOverride(
        "bmi",
        categorical=False,
        override_name="BMI",
        category=RowCategory.demographics,
        values_to_display=None,
        nonnormal=True,
    ),
    ColumnOverride(
        "cholesterol",
        categorical=False,
        override_name="Total cholesterol",
        category=RowCategory.lab_results,
        values_to_display=None,
    ),
    ColumnOverride(
        "kidney_failure",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.diagnoses,
    ),
    ColumnOverride(
        "angina",
        categorical=True,
        values_to_display=[1.0],
        override_name=None,
        category=RowCategory.diagnoses,
    ),
]


def table_one_facade(
    output_dir: Path,
    run: PsycopMlflowRun,
    sex_col_name: str = "pred_sex_female_fallback_0",
    overrides: Sequence[ColumnOverride] = cvd_overrides,
):
    model = table_one_model(run=run, sex_col_name=sex_col_name)
    view = table_one_view(model=model, overrides=overrides)
    view.to_csv(output_dir / "table_one.csv")


if __name__ == "__main__":
    table_one_facade(Path(), MlflowClientWrapper().get_run("CVD", "CVD 1, base, XGB"))
