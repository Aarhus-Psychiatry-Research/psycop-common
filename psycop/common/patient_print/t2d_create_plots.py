from pathlib import Path

import polars as pl

from psycop.common.patient_print.healthprints_config import HEALTHPRINTS_DATASETS_DIR
from psycop.common.patient_print.test_patient_printer import (
    HealthPrintPredictionTime,
    create_health_prints_from_patients,
)


def create_plots_from_dataset(dataset_path: Path, subtype: str):
    dataset = pl.read_parquet(dataset_path)

    massaged_ds = (
        dataset.with_columns(
            (
                (pl.col("event_timestamp") - pl.col("pred_timestamp")).dt.minutes()
                / (24 * 60)
            )
            .round(1)
            .alias("rel_time")
        )
        .select(["pred_time_uuid", "event_type", "event_value", "rel_time"])
        .rename({"event_type": "type", "event_value": "value"})
    )

    patient_dicts = massaged_ds.partition_by("pred_time_uuid", as_dict=True)

    patients = [
        HealthPrintPredictionTime(
            df=patient_dicts[k],
            i=i,
            x_max=-0,
            x_min=(-365 * 5),
            subtype=subtype,
            output_dir=Path(HEALTHPRINTS_DATASETS_DIR / "plots"),
        )
        for i, k in enumerate(patient_dicts)
    ]

    create_health_prints_from_patients(patients=patients)


if __name__ == "__main__":
    assert all(
        (HEALTHPRINTS_DATASETS_DIR / f"{subtype}.parquet").exists()
        for subtype in ("positive", "negative")
    )

    for subtype in ("negative", "positive"):
        create_plots_from_dataset(
            dataset_path=HEALTHPRINTS_DATASETS_DIR / f"{subtype}.parquet",
            subtype=subtype,
        )
